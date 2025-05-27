# test_pickle_stability_enhanced.py

import pickle
import hashlib
import sys
import math  # For inf, nan
import random
import string
from decimal import Decimal  # For precise float-like behavior if needed
import json  # For output
import os
import inspect
import ast
import coverage
import traceback
from collections import defaultdict
import importlib.util


# --- AST Visitor for Def-Use Analysis ---
class DefUseExtractor(ast.NodeVisitor):
    """
    Extracts variable definitions, uses, and def-use pairs from an AST.
    This provides a basis for "all-def" and "all-uses" white-box testing criteria.
    """

    def __init__(self):
        # Stores {var_name: [(lineno_def, scope_name), ...]}
        self.all_definitions = defaultdict(list)
        # Stores {var_name: [(lineno_use, scope_name), ...]}
        self.all_uses = defaultdict(list)
        # Stores list of {'variable': str, 'def_line': int, 'def_scope': str, 'use_line': int, 'use_scope': str}
        self.def_use_pairs = []

        self._current_function_name = None
        # Tracks the LATEST definition line for a variable in each scope.
        # Format: {scope_name: {var_name: def_lineno}}
        self._active_defs_in_scopes = {"__module__": {}}

    def _get_current_scope_name(self):
        """Returns the name of the current scope ('__module__' or function name)."""
        return self._current_function_name if self._current_function_name else "__module__"

    def visit_FunctionDef(self, node):
        old_function_name = self._current_function_name
        self._current_function_name = node.name

        # Functions define a new scope. Inherit/copy active defs from parent for closure,
        # or start fresh. For this analysis, we mostly care about defs within the func or module.
        # Initialize current function's active defs, potentially copying from module scope for globals.
        parent_scope_name = old_function_name if old_function_name else "__module__"
        # A simple model: function scope starts "empty" for its own active defs, but can see module's.
        self._active_defs_in_scopes[node.name] = {}

        current_scope_name = self._get_current_scope_name()

        # Arguments are definitions at the start of the function
        args_to_process = node.args.args
        if node.args.vararg:
            args_to_process.append(node.args.vararg)
        if node.args.kwarg:
            args_to_process.append(node.args.kwarg)
        if hasattr(node.args, 'posonlyargs'):  # Python 3.8+
            args_to_process.extend(node.args.posonlyargs)
        if hasattr(node.args, 'kwonlyargs'):  # Python 3.0+
            args_to_process.extend(node.args.kwonlyargs)

        for arg in args_to_process:
            if arg is None: continue  # Should not happen with standard ast.arguments
            arg_name = arg.arg
            self.all_definitions[arg_name].append((node.lineno, current_scope_name))
            # This argument definition is now active in this function's scope
            self._active_defs_in_scopes[current_scope_name][arg_name] = node.lineno

        self.generic_visit(node)  # Visit child nodes (body of the function)

        self._current_function_name = old_function_name  # Restore outer scope name

    def visit_Assign(self, node):
        # Process RHS (value) for uses first, as they occur before the LHS definition.
        if node.value:
            self.visit(node.value)

        current_scope_name = self._get_current_scope_name()
        # Ensure current scope exists in _active_defs_in_scopes
        if current_scope_name not in self._active_defs_in_scopes:
            self._active_defs_in_scopes[current_scope_name] = {}

        current_scope_active_defs = self._active_defs_in_scopes[current_scope_name]

        for target in node.targets:
            if isinstance(target, ast.Name):
                var_name = target.id
                self.all_definitions[var_name].append((node.lineno, current_scope_name))
                # This definition is now active in the current scope, overwriting any previous one.
                current_scope_active_defs[var_name] = node.lineno
            elif isinstance(target, (ast.Tuple, ast.List)):  # Unpacking assignment
                for elt in target.elts:
                    if isinstance(elt, ast.Name):
                        var_name = elt.id
                        self.all_definitions[var_name].append((node.lineno, current_scope_name))
                        current_scope_active_defs[var_name] = node.lineno
            # Not handling ast.Attribute targets (e.g., self.x = ...) as new "variable" definitions here,
            # but 'self' would be a use if 'self' is an ast.Name.
            # To handle those, one might track attributes on objects.

        # Do not call generic_visit(node) again if children (value, targets) were handled.
        # However, targets could be complex (e.g., subscripts), so parts of targets might need visiting if not ast.Name.
        # For simplicity, this focuses on ast.Name targets. If targets are ast.Attribute or ast.Subscript,
        # their .value (e.g., the object being indexed/attributed) is visited by generic_visit if we let it.
        # Let's ensure targets are processed for any nested uses if they are not simple Names.
        for target_node in node.targets:
            if not isinstance(target_node, ast.Name):
                self.visit(target_node)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):  # This is a variable use
            var_name = node.id
            use_line = node.lineno
            current_scope_name = self._get_current_scope_name()
            self.all_uses[var_name].append((use_line, current_scope_name))

            # Attempt to find the reaching definition for this use.
            # 1. Check current function's scope.
            active_def_line = self._active_defs_in_scopes.get(current_scope_name, {}).get(var_name)
            def_scope_of_active_def = current_scope_name if active_def_line is not None else None

            # 2. If not in local function scope and it's not the module scope itself, check module scope (for globals/closures).
            if active_def_line is None and current_scope_name != "__module__":
                active_def_line = self._active_defs_in_scopes.get("__module__", {}).get(var_name)
                def_scope_of_active_def = "__module__" if active_def_line is not None else None

            # 3. Python's LEGB rule also includes built-ins. This simplified analysis doesn't track built-ins.

            if active_def_line is not None and def_scope_of_active_def is not None:
                self.def_use_pairs.append({
                    'variable': var_name,
                    'def_line': active_def_line,
                    'def_scope': def_scope_of_active_def,
                    'use_line': use_line,
                    'use_scope': current_scope_name
                })
        # ast.Store is handled by visit_Assign, visit_For, visit_FunctionDef args, etc.
        # ast.Del would be a "kill" of a definition site.
        self.generic_visit(node)  # Continue traversal

    def visit_Import(self, node):
        current_scope_name = self._get_current_scope_name()
        if current_scope_name not in self._active_defs_in_scopes:  # Ensure scope dict exists
            self._active_defs_in_scopes[current_scope_name] = {}

        for alias in node.names:
            name_to_def = alias.asname if alias.asname else alias.name
            self.all_definitions[name_to_def].append((node.lineno, current_scope_name))
            self._active_defs_in_scopes[current_scope_name][name_to_def] = node.lineno
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        current_scope_name = self._get_current_scope_name()
        if current_scope_name not in self._active_defs_in_scopes:  # Ensure scope dict exists
            self._active_defs_in_scopes[current_scope_name] = {}

        for alias in node.names:
            name_to_def = alias.asname if alias.asname else alias.name
            self.all_definitions[name_to_def].append((node.lineno, current_scope_name))
            self._active_defs_in_scopes[current_scope_name][name_to_def] = node.lineno
        self.generic_visit(node)

    # Handling definitions in comprehensions and loops
    def visit_ListComp(self, node):
        self._visit_comprehension(node)

    def visit_SetComp(self, node):
        self._visit_comprehension(node)

    def visit_DictComp(self, node):
        self._visit_comprehension(node)

    def visit_GeneratorExp(self, node):
        self._visit_comprehension(node)

    def _visit_comprehension(self, node):
        # Comprehensions create a new scope in Python 3+
        # The target in each generator is a definition within that comprehension's scope.
        # This simplified model does not fully isolate comprehension scopes but registers defs.
        current_scope_name = self._get_current_scope_name()  # Or a special comp_scope
        if current_scope_name not in self._active_defs_in_scopes:
            self._active_defs_in_scopes[current_scope_name] = {}

        for generator in node.generators:
            self.visit(generator.iter)  # Visit the iterable (uses)
            # Target(s) in generator are definitions
            self._process_target_as_def(generator.target, generator.target.lineno, current_scope_name)
            for if_clause in generator.ifs:  # Conditions are uses
                self.visit(if_clause)

        # Visit the element/key/value expression
        if hasattr(node, 'elt'): self.visit(node.elt)
        if hasattr(node, 'key'): self.visit(node.key)  # For DictComp
        if hasattr(node, 'value'): self.visit(node.value)  # For DictComp

    def visit_For(self, node):
        current_scope_name = self._get_current_scope_name()
        if current_scope_name not in self._active_defs_in_scopes:
            self._active_defs_in_scopes[current_scope_name] = {}

        self.visit(node.iter)  # Iterable is a use
        # Target(s) in for loop are definitions
        self._process_target_as_def(node.target, node.target.lineno, current_scope_name)

        for body_item in node.body:
            self.visit(body_item)
        for orelse_item in node.orelse:
            self.visit(orelse_item)

    def _process_target_as_def(self, target_node, line_no, scope_name):
        """Helper to process assignment targets (Name, Tuple, List) as definitions."""
        if scope_name not in self._active_defs_in_scopes:  # Ensure scope dict exists
            self._active_defs_in_scopes[scope_name] = {}

        if isinstance(target_node, ast.Name):
            self.all_definitions[target_node.id].append((line_no, scope_name))
            self._active_defs_in_scopes[scope_name][target_node.id] = line_no
        elif isinstance(target_node, (ast.Tuple, ast.List)):
            for elt in target_node.elts:
                self._process_target_as_def(elt, line_no, scope_name)  # Recurse for nested structures

    def get_definitions_report(self):
        return self.all_definitions

    def get_uses_report(self):
        return self.all_uses

    def get_def_use_pairs_report(self):
        return sorted(self.def_use_pairs, key=lambda x: (x['def_scope'], x['variable'], x['def_line'], x['use_line']))


# --- White-box Testing Additions ---
class PickleWhiteBoxAnalyzer:
    """Analyzes pickle module source code for white-box testing"""

    def __init__(self):
        self.pickle_module_path = inspect.getfile(pickle)
        self.pickle_source = None
        self.ast_tree = None  # Store the AST tree
        self.function_defs_ast = {}  # Stores {func_name: {'lineno', 'args', 'body_size'}}
        self.control_flow_paths = []
        self.coverage_data = defaultdict(set)
        self.c_pickle_functions = {}  # For C implementation details

        # Attributes for def-use analysis
        self.all_definitions = defaultdict(list)  # {var_name: [(def_lineno, scope), ...]}
        self.all_uses = defaultdict(list)  # {var_name: [(use_lineno, scope), ...]}
        self.def_use_pairs = []  # List of {'variable': ..., 'def_line': ..., ...}

    def load_pickle_source(self):
        """Load pickle module source code and parse it into an AST."""
        try:
            with open(self.pickle_module_path, 'r', encoding='utf-8') as f:
                self.pickle_source = f.read()
            self.ast_tree = ast.parse(self.pickle_source)
            return True
        except Exception as e:
            print(f"Warning: Could not load or parse pickle source for AST analysis: {e}")
            # Try to load _pickle (C implementation) info if Python source fails
            self._load_c_pickle_info()
            return False

    def _load_c_pickle_info(self):
        """Load information about C pickle implementation if Python source is unavailable."""
        self.c_pickle_functions = {
            'Pickler': ['dump', 'clear_memo', 'persistent_id'],
            'Unpickler': ['load', 'find_class', 'persistent_load'],
            'dumps': 'Serialize obj to bytes',
            'loads': 'Deserialize bytes to obj',
            'HIGHEST_PROTOCOL': 'Highest protocol version',
            'DEFAULT_PROTOCOL': 'Default protocol version'
        }
        print("Note: White-box AST analysis for def-use chains requires Python source of 'pickle' module.")

    def analyze_source_structure(self):
        """Analyze pickle source code structure using AST for functions, CFG, and def-use."""
        if not self.ast_tree:
            print("Warning: AST tree not available for source structure analysis. Load source first.")
            return

        try:
            self._extract_ast_function_defs()
            self._extract_control_flow()

            # Perform def-use analysis
            extractor = DefUseExtractor()
            extractor.visit(self.ast_tree)
            self.all_definitions = extractor.get_definitions_report()
            self.all_uses = extractor.get_uses_report()
            self.def_use_pairs = extractor.get_def_use_pairs_report()

        except Exception as e:
            print(f"Warning: Could not fully analyze pickle source structure: {e}")
            traceback.print_exc()

    def _extract_ast_function_defs(self):
        """Extract all function definitions from the AST."""
        if not self.ast_tree: return
        self.function_defs_ast = {}
        for node in ast.walk(self.ast_tree):
            if isinstance(node, ast.FunctionDef):
                self.function_defs_ast[node.name] = {
                    'lineno': node.lineno,
                    'args': [arg.arg for arg in node.args.args],
                    'body_size': len(node.body)
                }

    def _extract_control_flow(self):
        """Extract control flow patterns (If, While, For) from the AST."""
        if not self.ast_tree: return
        self.control_flow_paths = []
        for node in ast.walk(self.ast_tree):
            if isinstance(node, (ast.If, ast.While, ast.For)):
                self.control_flow_paths.append({
                    'type': node.__class__.__name__,
                    'lineno': getattr(node, 'lineno', 0)
                })

    def get_critical_functions(self):
        """Identify critical pickle functions based on a predefined list and AST analysis."""
        critical_function_names = [
            'dump', 'dumps', 'load', 'loads',
            'Pickler', 'Unpickler',
            '_getattribute', 'whichmodule',  # These might be Python or C implemented
            'encode_long', 'decode_long'  # Often C implemented
        ]
        # If Python source was parsed, function_defs_ast will be populated.
        return {name: info for name, info in self.function_defs_ast.items()
                if name in critical_function_names}

    def generate_coverage_report(self):
        """Generate a line/branch coverage report for pickle module using the 'coverage' package."""
        cov = coverage.Coverage(source_pkgs=['pickle'])  # Focus on pickle package
        cov.start()

        test_objects = [
            42, "string", [1, 2, 3], {"key": "value"},
            float('inf'), None, True, b"bytes",
            MyObjectStable("cov_obj", 1)  # Use a custom object for better coverage
        ]
        for obj in test_objects:  # Run diverse operations
            for proto in range(-1, pickle.HIGHEST_PROTOCOL + 1):  # Test various protocols
                try:
                    pickled = pickle.dumps(obj, protocol=proto if proto >= 0 else None)
                    pickle.loads(pickled)
                except Exception:  # Ignore errors during coverage generation
                    pass
        try:
            # Test some Pickler/Unpickler specific paths
            from io import BytesIO
            f = BytesIO()
            p = pickle.Pickler(f, protocol=pickle.HIGHEST_PROTOCOL)
            p.dump(test_objects[0])
            p.dump(test_objects[1])
            f.seek(0)
            up = pickle.Unpickler(f)
            up.load()
            up.load()
        except Exception:
            pass

        cov.stop()
        coverage_summary = {}
        try:
            # Using a temporary directory for report files if html/xml reports are needed
            # report_dir = "coverage_report_pickle"
            # os.makedirs(report_dir, exist_ok=True)
            # cov.html_report(directory=report_dir)
            # print(f"HTML coverage report generated in {report_dir}")

            # For JSON output, get data directly
            cov_data = cov.get_data()
            if cov_data:
                # Try to find the main pickle.py file if source was loaded
                pickle_py_file = self.pickle_module_path if self.pickle_source else None

                for file_path in cov_data.measured_files():
                    # Focus on the pickle.py we are analyzing, or any file related to 'pickle'
                    is_target_file = (pickle_py_file and os.path.samefile(file_path, pickle_py_file)) or \
                                     ('pickle' in os.path.basename(file_path).lower() and file_path.endswith('.py'))

                    if is_target_file:
                        analysis = cov.analysis2(
                            file_path)  # (filename, statements, excluded, missing, missing_formatted)
                        total_statements = len(analysis[1])
                        executed_statements = total_statements - len(analysis[3])
                        percent_covered = (executed_statements / total_statements * 100) if total_statements > 0 else 0

                        coverage_summary[os.path.basename(file_path)] = {
                            'path': file_path,
                            'total_statements': total_statements,
                            'executed_statements': executed_statements,
                            'missing_statements': len(analysis[3]),
                            'percent_covered': f"{percent_covered:.2f}%",
                            'sample_missing_lines': sorted(list(analysis[3]))[:20]  # First 20 missing lines
                        }
            if not coverage_summary:
                coverage_summary[
                    'info'] = "No specific pickle.py coverage data captured. Check source_pkgs or if pickle is C-implemented."

        except Exception as e:
            return {'error': f"Failed to generate coverage report: {e}"}
        finally:
            cov.erase()  # Clean up .coverage data file
        return coverage_summary

    # --- Reports for Def-Use Analysis ---
    def get_definitions_report(self):
        """Returns a structured report of all variable definitions found."""
        return self.all_definitions

    def get_uses_report(self):
        """Returns a structured report of all variable uses found."""
        return self.all_uses

    def get_def_use_pairs_report(self):
        """Returns a structured report of def-use pairs."""
        return self.def_use_pairs


# --- Path Coverage Testing ---
class PicklePathCoverageTester:
    """Tests specific code paths in pickle module"""

    def __init__(self):
        self.path_test_cases = []
        self._generate_path_specific_tests()

    def _generate_path_specific_tests(self):
        """Generate test cases for specific code paths"""

        # Test memo optimization paths
        self.path_test_cases.append({
            'id': 'path_memo_reuse',
            'description': 'Test pickle memo reuse for identical objects',
            'generator': self._create_memo_reuse_case
        })

        # Test protocol-specific paths
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            self.path_test_cases.append({
                'id': f'path_protocol_{protocol}_specific',
                'description': f'Test protocol {protocol} specific code paths',
                'generator': lambda p=protocol: self._create_protocol_specific_case(p)
            })

        # Test error handling paths
        self.path_test_cases.append({
            'id': 'path_unpickleable_object',
            'description': 'Test error handling for unpickleable objects',
            'generator': self._create_unpickleable_case
        })

        # Test recursive depth limits
        self.path_test_cases.append({
            'id': 'path_recursion_limit',
            'description': 'Test recursion depth limit handling',
            'generator': self._create_deep_recursion_case
        })

        # Test large object handling
        self.path_test_cases.append({
            'id': 'path_large_object',
            'description': 'Test large object serialization paths',
            'generator': self._create_large_object_case
        })

    def _create_memo_reuse_case(self):
        """Create test case for memo reuse"""
        shared_list = [1, 2, 3]
        return {
            'data': [shared_list, shared_list, {'key': shared_list}],
            'expected_behavior': 'Should reuse memo for identical objects'
        }

    def _create_protocol_specific_case(self, protocol):
        """Create protocol-specific test cases"""
        # Different protocols handle certain objects differently
        if protocol < 2:
            # Old protocols don't support new-style classes well
            return {'data': type('OldProtocolTest', (), {'value': 42})()}
        elif protocol >= 4:
            # Protocol 4+ supports more efficient serialization
            return {'data': bytes(range(256)) * 100}  # Large bytes object
        else:
            return {'data': set(range(100))}  # Sets require protocol 2+

    def _create_unpickleable_case(self):
        """Create unpickleable object to test error paths"""
        import threading
        return {'data': threading.Lock(), 'expect_error': True}

    def _create_deep_recursion_case(self):
        """Create deeply recursive structure"""
        # Reduced depth to be less likely to hit system recursion limits during test generation itself
        depth = sys.getrecursionlimit() // 10
        if depth > 150: depth = 150  # Cap depth for test stability
        obj = []
        current = obj
        for _ in range(depth):
            new_list = []
            current.append(new_list)
            current = new_list
        return {'data': obj}

    def _create_large_object_case(self):
        """Create large object to test size handling"""
        # Large string to test string optimization paths
        return {'data': 'x' * (10 ** 5)}  # 100KB string (reduced from 1MB to speed up tests)

    def get_test_cases(self):
        """Get all path-specific test cases"""
        cases = []
        for path_test in self.path_test_cases:
            try:
                test_data_details = path_test['generator']()
                cases.append({
                    'id': path_test['id'],
                    'data': test_data_details.get('data'),
                    'desc': path_test['description'],
                    'expect_error': test_data_details.get('expect_error', False)
                })
            except Exception as e:
                cases.append({
                    'id': path_test['id'],
                    'data': None,
                    'desc': f"{path_test['description']} (Generation failed: {e})",
                    'expect_error': True  # Assume error if generation fails
                })
        return cases


# --- Traceability Matrix ---
class TraceabilityMatrix:
    """Maps pickle features to test cases"""

    def __init__(self):
        self.feature_map = {
            'Basic Types': {
                'integers': ['int_simple', 'int_zero', 'int_neg', 'int_large_pos'],
                'floats': ['float_simple', 'float_inf_pos', 'float_nan'],
                'strings': ['string_simple', 'string_unicode', 'string_empty'],
                'bytes': ['bytes_simple', 'bytes_with_null'],
                'booleans': ['boolean_true', 'boolean_false'],
                'none': ['none_type']
            },
            'Container Types': {
                'lists': ['list_simple', 'list_empty', 'list_many_elements_homo'],
                'tuples': ['tuple_simple', 'tuple_empty', 'tuple_single_element'],
                'dicts': ['dict_simple', 'dict_ordered_keys_insertion'],
                'sets': ['set_simple', 'set_with_special_floats'],
                'frozensets': ['frozenset_simple', 'frozenset_empty']
            },
            'Protocol Features': {
                'protocol_0': ['path_protocol_0_specific', 'All tests with Protocol_0'],
                'protocol_1': ['path_protocol_1_specific', 'All tests with Protocol_1'],
                'protocol_2': ['path_protocol_2_specific', 'set_*', 'frozenset_*'],
                'protocol_3': ['path_protocol_3_specific', 'bytes_simple', 'bytes_with_null'],
                'protocol_4': ['path_protocol_4_specific', 'All tests with Protocol_4'],
                'protocol_5': ['path_protocol_5_specific', 'All tests with Protocol_5 if available']
            },
            'Special Methods': {
                '__reduce__': ['reducible_point'],
                '__reduce_ex__': ['reducible_point_ex'],
                '__getstate__/__setstate__': ['custom_obj_stable_getstate'],
                'memo_usage': ['memo_tricky_list_with_shared', 'path_memo_reuse']
            },
            'Error Handling': {
                'unpickleable_objects': ['path_unpickleable_object'],
                'recursion_limits': ['path_recursion_limit'],
                'corrupt_data': ['Manual testing required']  # Corrupt data testing is typically dynamic
            },
            'Performance Paths': {  # Paths that might stress performance or specific optimizations
                'large_objects': ['path_large_object', 'string_long'],
                'deep_nesting': ['list_deeply_nested_simple', 'path_recursion_limit'],
                # path_recursion_limit creates deep nesting
                'many_references': ['structure_with_shared_obj', 'path_memo_reuse']
            }
        }

    def generate_matrix_report(self, test_results):
        """Generate traceability matrix report based on executed test case IDs."""
        report = {
            'feature_coverage': {},
            'test_to_feature_mapping': defaultdict(list),  # test_id -> [feature1, feature2]
            'uncovered_features': [],
            'notes': []
        }

        all_run_test_ids = set()
        for category_results in test_results.values():
            all_run_test_ids.update(category_results.keys())

        for feature_category, features in self.feature_map.items():
            report['feature_coverage'][feature_category] = {}
            for feature, associated_test_patterns in features.items():
                covered_by_tests = set()
                is_fully_covered = True  # Assume fully covered unless a pattern maps to no run tests

                for pattern in associated_test_patterns:
                    matched_one_test_for_pattern = False
                    # '*' means "any test matching the category implicitly" or wildcard match
                    if pattern.startswith("All tests with Protocol_"):  # Special handling
                        # This implies a dynamic check based on protocol used in test runs,
                        # For simplicity, we assume if protocol specific path tests run, it's covered.
                        # e.g. if 'path_protocol_0_specific' ran.
                        proto_test_id = pattern.replace("All tests with ", "path_").replace(" if available",
                                                                                            "") + "_specific"
                        if proto_test_id in all_run_test_ids:
                            covered_by_tests.add(proto_test_id)
                            report['test_to_feature_mapping'][proto_test_id].append(f"{feature_category}.{feature}")
                            matched_one_test_for_pattern = True
                        else:
                            # If a specific protocol path test didn't run (e.g. protocol 5 not available)
                            # this doesn't mean the feature is "uncovered" if the protocol isn't applicable.
                            # This logic could be more sophisticated.
                            pass  # Don't mark as uncovered if path test for that proto didn't exist/run.

                    elif pattern.endswith("*"):  # Wildcard like 'set_*'
                        prefix = pattern[:-1]
                        for test_id in all_run_test_ids:
                            if test_id.startswith(prefix):
                                covered_by_tests.add(test_id)
                                report['test_to_feature_mapping'][test_id].append(f"{feature_category}.{feature}")
                                matched_one_test_for_pattern = True
                    else:  # Direct test ID
                        if pattern in all_run_test_ids:
                            covered_by_tests.add(pattern)
                            report['test_to_feature_mapping'][pattern].append(f"{feature_category}.{feature}")
                            matched_one_test_for_pattern = True

                    if not matched_one_test_for_pattern and not pattern.startswith("All tests with Protocol_"):
                        # Only mark as not fully covered if a non-protocol-availability pattern found no tests
                        if pattern != "Manual testing required":  # Manual tests are out of scope for auto-check
                            is_fully_covered = False

                # Coverage percentage is a bit tricky with patterns.
                # Let's report if *any* associated test ran.
                coverage_status = "Covered" if covered_by_tests else "Not Covered"
                if not is_fully_covered and pattern != "Manual testing required":
                    coverage_status = "Partially Covered / Check Patterns"

                report['feature_coverage'][feature_category][feature] = {
                    'status': coverage_status,
                    'associated_test_patterns': associated_test_patterns,
                    'covering_test_ids': sorted(list(covered_by_tests)),
                    'notes': "Manual testing required" if "Manual testing required" in associated_test_patterns else ""
                }

                if coverage_status != "Covered" and "Manual testing required" not in associated_test_patterns:
                    report['uncovered_features'].append(f"{feature_category}.{feature}")

        # Clean up test_to_feature_mapping
        for test_id in report['test_to_feature_mapping']:
            report['test_to_feature_mapping'][test_id] = sorted(list(set(report['test_to_feature_mapping'][test_id])))

        return report

    def _is_test_in_results(self, test_id_pattern, test_results_keys):
        """Check if a test ID pattern matches any key in results."""
        if test_id_pattern.endswith('*'):
            prefix = test_id_pattern[:-1]
            for key in test_results_keys:
                if key.startswith(prefix):
                    return True
            return False
        return test_id_pattern in test_results_keys


# --- Helper Functions ---
def get_pickle_hash(data_object, protocol=None, note=""):
    """
    Pickles an object and returns its SHA256 hash.
    Args:
        data_object: The Python object to pickle.
        protocol: The pickle protocol to use. None means default.
        note: A small note to print, e.g. about the object.
    Returns:
        A hex string representing the SHA256 hash of the pickled object
        or an error message string.
    """
    try:
        pickled_data = pickle.dumps(data_object, protocol=protocol)
        hasher = hashlib.sha256()
        hasher.update(pickled_data)
        hex_digest = hasher.hexdigest()
        return hex_digest
    except Exception as e:
        error_message = (
            f"Error pickling object ({note}) with protocol "
            f"{protocol if protocol is not None else 'Default (' + str(pickle.DEFAULT_PROTOCOL) + ')'}: {type(e).__name__} - {e}"
        )  # Added type(e).__name__
        return f"ERROR: {error_message}"


def get_limited_repr(data_object, max_chars=2048):
    """
    Returns repr(data_object), truncated if it exceeds max_chars.
    Args:
        data_object: The Python object to represent.
        max_chars: The maximum number of characters for the representation.
    Returns:
        A string representation of the data_object.
    """
    try:
        r = repr(data_object)
    except Exception as e:
        # Fallback if repr() itself fails on a strange object
        return f"<Error generating repr: {type(data_object).__name__} - {e}>"
    if len(r) > max_chars:
        return r[:max_chars - 3] + "..."
    return r


# --- Test Cases ---

# 1. Basic Data Types (Extended)
test_cases_basic = [
    {"id": "int_simple", "data": 123, "desc": "A simple integer"},
    {"id": "int_zero", "data": 0, "desc": "Integer zero"},
    {"id": "int_neg", "data": -42, "desc": "A negative integer"},
    {"id": "int_small_neg", "data": -1, "desc": "Small negative integer"},
    {
        "id": "int_large_pos",
        "data": 2 ** 100,
        "desc": "A large positive integer (Python arbitrary precision)",
    },
    {"id": "int_large_neg", "data": -(2 ** 100), "desc": "A large negative integer"},
    {
        "id": "int_boundary_32bit_pos",
        "data": 2 ** 31 - 1,
        "desc": "Max positive 32-bit signed int",
    },
    {
        "id": "int_boundary_32bit_plus_one",
        "data": 2 ** 31,
        "desc": "Max positive 32-bit signed int + 1",
    },
    {"id": "float_simple", "data": 3.14159, "desc": "A simple float"},
    {"id": "float_zero", "data": 0.0, "desc": "Float zero"},
    {
        "id": "float_neg_zero",
        "data": -0.0,
        "desc": "Float negative zero (note: -0.0 == 0.0 is True)",
    },
    {"id": "float_small_pos", "data": 1e-10, "desc": "Small positive float"},
    {"id": "float_small_neg", "data": -1e-10, "desc": "Small negative float"},
    {
        "id": "float_large_pos",
        "data": 1.79e308,
        "desc": "Large positive float (near double precision max)",
    },
    {"id": "float_large_neg", "data": -1.79e308, "desc": "Large negative float"},
    {"id": "float_sci", "data": 1.23e45, "desc": "Float in scientific notation"},
    {
        "id": "float_calc_sum",
        "data": 0.1 + 0.2,
        "desc": "Float from sum 0.1 + 0.2 (precision issues known for value)",
    },
    {"id": "float_inf_pos", "data": float("inf"), "desc": "Positive infinity"},
    {"id": "float_inf_neg", "data": float("-inf"), "desc": "Negative infinity"},
    {"id": "float_nan", "data": float("nan"), "desc": "Not a Number (NaN)"},
    {"id": "string_simple", "data": "hello world", "desc": "A simple ASCII string"},
    {"id": "string_empty", "data": "", "desc": "An empty string"},
    {
        "id": "string_unicode",
        "data": "你好世界🌍 éàçöüß",
        "desc": "A Unicode string with various chars",
    },
    {"id": "string_long", "data": "a" * 1000, "desc": "A long string"},
    {
        "id": "string_whitespace",
        "data": "  leading and trailing spaces  \n\t tabs and newlines ",
        "desc": "String with various whitespace",
    },
    {
        "id": "string_looks_like_num",
        "data": "123.456",
        "desc": "String that looks like a number",
    },
    {
        "id": "string_pickle_chars",
        "data": ".\n()[]{}\"'\\;",
        "desc": "String with potential pickle control/special chars",
    },
    {"id": "bytes_simple", "data": b"hello", "desc": "Simple bytestring"},
    {"id": "bytes_empty", "data": b"", "desc": "Empty bytestring"},
    {"id": "bytes_with_null", "data": b"he\x00llo", "desc": "Bytestring with null char"},
    {"id": "boolean_true", "data": True, "desc": "Boolean True"},
    {"id": "boolean_false", "data": False, "desc": "Boolean False"},
    {"id": "none_type", "data": None, "desc": "NoneType object"},
]

# 2. Container Types (Extended)
test_cases_containers = [
    {
        "id": "list_simple",
        "data": [1, "two", 3.0, None, True],
        "desc": "Simple list with mixed types",
    },
    {"id": "list_empty", "data": [], "desc": "Empty list"},
    {
        "id": "list_single_element",
        "data": ["lonely"],
        "desc": "List with a single element",
    },
    {
        "id": "list_many_elements_homo",
        "data": list(range(100)),
        "desc": "List with many homogenous elements",
    },
    {
        "id": "list_many_elements_hetero",
        "data": [i if i % 2 == 0 else str(i) for i in range(100)],
        "desc": "List with many heterogenous elements",
    },
    {
        "id": "list_with_special_floats",
        "data": [0.0, -0.0, float("inf"), float("-inf"), float("nan")],
        "desc": "List containing special floats",
    },
    {"id": "tuple_simple", "data": (1, "two", 3.0, None, False), "desc": "Simple tuple"},
    {"id": "tuple_empty", "data": (), "desc": "Empty tuple"},
    {
        "id": "tuple_single_element",
        "data": ("single",),
        "desc": "Tuple with a single element (note comma)",
    },
    {
        "id": "tuple_many_elements",
        "data": tuple(range(50)),
        "desc": "Tuple with many elements",
    },
    {
        "id": "dict_simple",
        "data": {"a": 1, "b": "two", "c": 3.0, "d": None, "e": True},
        "desc": "Simple dictionary",
    },
    {"id": "dict_empty", "data": {}, "desc": "Empty dictionary"},
    {
        "id": "dict_single_entry",
        "data": {"key": "value"},
        "desc": "Dictionary with a single entry",
    },
    {
        "id": "dict_numeric_keys",
        "data": {1: "one", 0: "zero", -1: "minus_one"},
        "desc": "Dictionary with numeric keys",
    },
    {
        "id": "dict_mixed_keys",
        "data": {1: "int_key", "str_key": 2, (1, 2): "tuple_key", True: "bool_key", None: "none_key"},
        # Added more key types
        "desc": "Dictionary with mixed key types",
    },
    {
        "id": "dict_ordered_keys_insertion",  # Python 3.7+ dicts are ordered. Pickle should preserve this.
        "data": {"z": 1, "y": 2, "x": 3, "a": 4, "b": 5, "c": 6},
        "desc": "Dict with specific insertion order (Python 3.7+ preserves it)",
    },
    {
        "id": "set_simple",
        "data": {1, "two", 3.0, None, True},  # Order of elements in set literal doesn't guarantee storage/pickle order
        "desc": "Simple set (EXPECT POTENTIAL VARIATION for protocols < 4 or across Pythons due to hash randomization)",
    },
    {"id": "set_empty", "data": set(), "desc": "Empty set"},
    {
        "id": "set_single_element",
        "data": {"hello"},
        "desc": "Set with a single element",
    },
    {
        "id": "set_all_same_type",
        "data": {10, 20, 30, 1, 2, 3},  # Order not guaranteed
        "desc": "Set with elements of same type (EXPECT POTENTIAL VARIATION)",
    },
    {
        "id": "set_with_special_floats",  # NaN behavior in sets is tricky
        "data": {0.0, -0.0, float("inf"), float("-inf"), float("nan"), float("nan")},  # two nans, but set stores one
        "desc": "Set containing special floats (NaN makes it tricky, EXPECT POTENTIAL VARIATION)",
    },
    {
        "id": "set_from_list_with_duplicates",
        "data": set([1, 2, 2, 3, 3, 3, 4, 1]),  # Result is {1,2,3,4}
        "desc": "Set created from list with duplicates (EXPECT POTENTIAL VARIATION)",
    },
    {
        "id": "frozenset_simple",
        "data": frozenset({1, "two", 3.0, None}),
        "desc": "Simple frozenset (EXPECT POTENTIAL VARIATION like sets)",
    },
    {"id": "frozenset_empty", "data": frozenset(), "desc": "Empty frozenset"},
]

# 3. Nested and Complex Structures (Extended)
deep_list_orig_logic = [1]
temp_list_for_deep_list = deep_list_orig_logic
# Reduced depth for faster test execution & lower recursion risk during test setup
for _ in range(5):  # Was 10
    new_level = [temp_list_for_deep_list]
    temp_list_for_deep_list.append(new_level)
    temp_list_for_deep_list = new_level
temp_list_for_deep_list.append("end_orig_deep")

simple_deep_item = "end_simple_deep"
for _ in range(15):  # Was 30
    simple_deep_item = [simple_deep_item]
simple_deep_list_data = simple_deep_item

complex_shared_obj = ["shared_part", 123]
structure_with_shared = [
    complex_shared_obj,
    complex_shared_obj,
    {"key1": complex_shared_obj, "key2": [complex_shared_obj, "other"]},
]

test_cases_complex = [
    {
        "id": "list_nested_basic",
        "data": [1, [2, 3], [4, [5, "six", {"deep_key": None}]]],
        "desc": "Basic nested list with mixed types",
    },
    {
        "id": "list_deeply_nested_orig_logic",
        "data": deep_list_orig_logic,
        "desc": "Deeply nested list using original recursive logic (depth approx 5)",
    },
    {
        "id": "list_deeply_nested_simple",
        "data": simple_deep_list_data,
        "desc": "Deeply nested list (non-recursive def, depth 15)",
    },
    {
        "id": "dict_nested_basic",
        "data": {"k1": "v1", "k2": [1, 2, {"sk1": "sv1", "sk2": [None, True]}], "k3": {},
                 "k4": ("t1", {"stk": b"bytes"})},
        "desc": "Basic nested dictionary with various types",
    },
    {
        "id": "list_of_dicts_varied",
        "data": [
            {"id": 1, "val": "a", "active": True, "tags": {"tag1", "tag2"}},
            {"id": 2, "val": None, "tags": ["x", "y"], "meta": {"source": "test"}},
        ],
        "desc": "List of dictionaries with varied content (EXPECT POTENTIAL VARIATION for sets)",
    },
    {
        "id": "dict_with_list_of_sets",
        "data": {"data_sets": [{1, 2}, {2, 3, 4}, frozenset({5, 6}), set()]},  # Added empty set
        "desc": "Dict containing list of sets/frozensets (EXPECT POTENTIAL VARIATION for sets)",
    },
    {
        "id": "structure_with_shared_obj",  # Tests memoization
        "data": structure_with_shared,
        "desc": "Structure with internally shared object reference (tests memoization)",
    },
    {
        "id": "list_containing_all_special_floats",  # Specific order for direct comparison
        "data": [float("inf"), float("-inf"), float("nan"), -0.0, 0.0, 1.0 / 3.0],
        "desc": "List with all special floats in specific order and a repeating decimal",
    },
    {
        "id": "set_containing_tuples_with_nans",  # Tuples are hashable even with NaN
        "data": {(1, float("nan")), (2, 0.0), (1, float("nan")), (float("nan"), 1)},  # (1, nan) is one element
        "desc": "Set of tuples where tuples contain NaN (complex comparison, EXPECT POTENTIAL VARIATION)",
    },
]

# 4. Recursive Structures
simple_recursive_list_data = []
simple_recursive_list_data.append(simple_recursive_list_data)  # l = [l]

simple_recursive_dict_data = {}
simple_recursive_dict_data["self"] = simple_recursive_dict_data  # d = {'self': d}

recursive_list_A_data = ["A_starts_here"]
recursive_list_B_data = ["B_starts_here"]
recursive_list_A_data.append(recursive_list_B_data)  # A = [..., B]
recursive_list_B_data.append(recursive_list_A_data)  # B = [..., A]

test_cases_recursive = [
    {
        "id": "list_recursive_direct",
        "data": simple_recursive_list_data,
        "desc": "Directly recursive list (l=[l])",
    },
    {
        "id": "dict_recursive_direct",
        "data": simple_recursive_dict_data,
        "desc": "Directly recursive dictionary (d={self:d})",
    },
    {
        "id": "list_AB_recursion",  # Mutually recursive lists
        "data": recursive_list_A_data,
        "desc": "Mutually recursive lists A <-> B",
    },
    {  # Test if memoization handles shared parts within recursion
        "id": "obj_indirect_recursion_complex_shared_part",
        "data": structure_with_shared[2],  # This is {"key1": complex_shared_obj, "key2": [complex_shared_obj, "other"]}
        # where complex_shared_obj is already shared. No direct recursion here unless complex_shared_obj becomes recursive.
        # This tests memoization of a shared non-recursive component.
        # For actual recursion involving this part, it would need to be modified.
        # Let's make it recursive for a better test.
        "desc": "Shared object part of a complex structure (tests memoization, not necessarily recursion unless data is made so)",
    },
]


# Make the shared part recursive for a stronger test
# complex_shared_obj.append(structure_with_shared) # This makes it structure_with_shared[0] -> ... -> structure_with_shared
# Be careful with modifying shared structures globally if tests are not isolated.
# For this script, test cases are generated once.

# 5. Custom Objects (Extended to show __getstate__/__setstate__ impact)
class MyObjectStable:
    def __init__(self, name, value, items_set=None):
        self.name = name
        self.value = value
        # Ensure set items are comparable for sorting, e.g. all strings or all numbers
        self.items_set = items_set if items_set is not None else set()

    def __getstate__(self):
        # Called during pickling. Should return a picklable state.
        state = self.__dict__.copy()
        # Ensure stability for sets by sorting them if possible.
        try:
            # Attempt to sort directly if elements are comparable
            state["items_set"] = sorted(list(self.items_set))
        except TypeError:
            # Fallback for non-sortable mixed types: sort by repr (still better than random)
            state["items_set"] = sorted(list(map(repr, self.items_set)))
        return state

    def __setstate__(self, state):
        # Called during unpickling. `state` is what __getstate__ returned.
        state["items_set"] = set(state["items_set"])  # Convert sorted list back to set
        self.__dict__.update(state)

    def __str__(self):  # For readable descriptions
        items_repr_list = []
        try:
            items_repr_list = sorted(list(self.items_set))
        except TypeError:
            items_repr_list = sorted(list(map(repr, self.items_set)))
        return (
            f"MyObjectStable(name='{self.name}', value={self.value}, "
            f"items_set={items_repr_list})"  # Show sorted list for stable string
        )

    def __repr__(self):  # For detailed representation, e.g. in outputs
        # For repr, it's conventional to show how to reconstruct the object.
        # Here, using the actual set for accuracy.
        return f"MyObjectStable({self.name!r}, {self.value!r}, items_set={self.items_set!r})"


class MyObjectUnstable:  # No __getstate__/__setstate__, relies on default pickling of __dict__
    def __init__(self, name, value, items_set=None):
        self.name = name
        self.value = value
        self.items_set = items_set if items_set is not None else set()  # Sets in __dict__ can cause variation

    def __str__(self):
        return (
            f"MyObjectUnstable(name='{self.name}', value={self.value}, "
            f"items_set={self.items_set})"  # Set string representation can vary
        )

    def __repr__(self):
        return f"MyObjectUnstable({self.name!r}, {self.value!r}, items_set={self.items_set!r})"


custom_obj_stable = MyObjectStable("stable_obj", 100, {"gamma_item", "alpha_item", "beta_item"})
custom_obj_unstable = MyObjectUnstable(
    "unstable_obj", 200, {"zeta_item", "epsilon_item", "delta_item"}  # Elements are strings, hash order might vary
)

# Dynamically created class
MyObjectDefault = type(
    "MyObjectDefault",  # Class name
    (object,),  # Base classes
    {  # Attributes and methods
        "__init__": lambda self, x_val, y_set_val: setattr(self, "attr1", x_val) or \
                                                   setattr(self, "attr2_set",
                                                           y_set_val if y_set_val is not None else set()),
        "attr1": "default_class_attr_val1",  # Class attribute
        "attr2_set": {"default_class_set_item"},  # Class attribute (set)
        "__str__": lambda self: (  # Instance method
            f"MyObjectDefault(attr1='{getattr(self, 'attr1', 'N/A')}', attr2_set={getattr(self, 'attr2_set', set())})"
        ),
        "__repr__": lambda self: (  # Instance method
            f"MyObjectDefault({getattr(self, 'attr1', 'N/A')!r}, {getattr(self, 'attr2_set', set())!r})"
        )
    },
)
custom_obj_default_instance = MyObjectDefault("instance_val1", {"data_set_inst_A", "data_set_inst_B"})

test_cases_custom_objects = [
    {
        "id": "custom_obj_stable_getstate",
        "data": custom_obj_stable,
        "desc": str(custom_obj_stable),  # str uses sorted list for items_set
    },
    {
        "id": "custom_obj_unstable_no_getstate_with_set_attr",
        "data": custom_obj_unstable,
        "desc": str(custom_obj_unstable) + " (EXPECT POTENTIAL VARIATION due to set in __dict__)",
    },
    {
        "id": "custom_obj_default_with_set_attr",  # Dynamically created class
        "data": custom_obj_default_instance,
        "desc": f"{str(custom_obj_default_instance)} (EXPECT POTENTIAL VARIATION for set in instance of dynamic class)",
    },
]


# 6. Advanced Pickling Control / White-box Inspired
class ReduciblePoint:  # Implements __reduce__
    def __init__(self, x_coord, y_coord):
        self.x = x_coord
        self.y = y_coord

    def __reduce__(self):
        # Returns (callable_to_reconstruct, args_for_callable, optional_state_to_pass_to_setstate)
        return (self.__class__, (self.x, self.y))  # Simple reconstruction

    def __str__(self): return f"ReduciblePoint({self.x}, {self.y})"

    def __repr__(self): return f"ReduciblePoint({self.x!r}, {self.y!r})"


class ReduciblePointEx(ReduciblePoint):  # Implements __reduce_ex__ for protocol-specific reduction
    def __init__(self, x_coord, y_coord, z_coord=0):
        super().__init__(x_coord, y_coord)
        self.z = z_coord
        self.extra_state_info = {"info": "from_reduce_ex_default", "protocol_version_at_pickle": -1}

    def __reduce_ex__(self, protocol_version):  # Takes protocol version
        self.extra_state_info["protocol_version_at_pickle"] = protocol_version
        # Returns (callable, args, state, listitems_iterator, dictitems_iterator, newobj_func_for_slots)
        return (
            self.__class__,  # Callable to create initial object
            (self.x, self.y, self.z),  # Args for the callable
            self.extra_state_info,  # State to be passed to __setstate__ (if any)
            None,  # Iterator for list items (if obj is list subclass)
            None  # Iterator for dict items (if obj is dict subclass)
        )

    def __setstate__(self, state_dict):  # To receive state from __reduce__ or __reduce_ex__
        self.extra_state_info = state_dict
        # If __init__ wasn't called by pickle (e.g. if __reduce__ returned a factory not the class)
        # ensure all instance attrs are set up. Here, __init__ is called by __reduce_ex__.

    def __str__(self):
        return (
            f"ReduciblePointEx({self.x}, {self.y}, {self.z}, "
            f"extra_state={self.extra_state_info})"
        )

    def __repr__(self):
        return f"ReduciblePointEx({self.x!r}, {self.y!r}, {self.z!r}, extra_state_info={self.extra_state_info!r})"


class MemoTricky:  # For testing memoization with __new__ and shared instances
    _instance_cache = {}  # Cache for flyweight pattern

    def __new__(cls, name_key):  # Controls object creation
        if name_key not in cls._instance_cache:
            instance = super().__new__(cls)
            instance.name_key = name_key  # Initialize *before* caching
            instance.data_list = []  # Initialize *before* caching
            instance.initialized_by_init_flag = False  # Flag for __init__
            cls._instance_cache[name_key] = instance
        return cls._instance_cache[name_key]

    def __init__(self, name_key):  # Called after __new__
        # Ensure __init__ logic runs only once per actual instance if __new__ returns existing one
        if not self.initialized_by_init_flag:
            # self.name_key = name_key # Already set in __new__ for cached instances
            # self.data_list = []    # Already set in __new__
            self.initialized_by_init_flag = True
            # print(f"MemoTricky __init__ called for {self.name_key}")

    def add_data(self, item_val):
        self.data_list.append(item_val)

    def __str__(self):
        return f"MemoTricky(name_key='{self.name_key}', data_list={self.data_list})"

    def __repr__(self):
        return f"MemoTricky({self.name_key!r})"  # Keep repr simple

    # For pickling, ensure state is captured correctly for __new__ pattern
    def __getstate__(self):
        return {"name_key": self.name_key, "data_list": self.data_list,
                "initialized_by_init_flag": self.initialized_by_init_flag}

    def __setstate__(self, state):
        # self.name_key = state["name_key"] # Should be handled by __new__ if unpickling creates via __new__
        self.data_list = state["data_list"]
        self.initialized_by_init_flag = state["initialized_by_init_flag"]
        # Note: Pickle calls __new__ then __setstate__ (if available), usually not __init__ on unpickle.
        # If __init__ is essential after unpickling, __setstate__ might need to call parts of it.


MemoTricky._instance_cache.clear()  # Clear cache before creating test instances
mt_shared_A = MemoTricky("shared_A_instance")
mt_shared_A.add_data(10)
mt_shared_A.add_data(20)
mt_unique_B = MemoTricky("unique_B_instance")  # Different key, new instance
mt_unique_B.add_data(1000)
# List containing shared and unique instances
memo_test_list_data = [mt_shared_A, mt_unique_B, mt_shared_A,
                       MemoTricky("another_A_instance")]  # Third is same as first

test_cases_advanced_pickling = [
    {
        "id": "reducible_point",  # Uses __reduce__
        "data": ReduciblePoint(15, 25),
        "desc": str(ReduciblePoint(15, 25)),
    },
    {
        "id": "reducible_point_ex",  # Uses __reduce_ex__
        "data": ReduciblePointEx(35, 45, 55),
        "desc": str(ReduciblePointEx(35, 45, 55)),
    },
    {
        "id": "memo_tricky_list_with_shared",  # Tests memoization with __new__
        "data": memo_test_list_data,
        "desc": f"List with shared MemoTricky instances: {[str(x) for x in memo_test_list_data]}",
    },
]
MemoTricky._instance_cache.clear()  # Clean up cache after test case generation

# 7. Fuzzing Test Cases
FUZZ_NUM_TEST_CASES = 20  # Reduced for faster execution
FUZZ_MAX_DEPTH = 2  # Reduced depth
FUZZ_MAX_COLLECTION_SIZE = 3  # Reduced size
FUZZ_MAX_STRING_LENGTH = 30
FUZZ_MAX_BYTES_LENGTH = 30
FUZZ_MAX_INT = 2 ** 20  # Smaller int range
FUZZ_MIN_INT = -(2 ** 20)
FUZZ_SEED = 42  # Fixed seed for reproducibility
random.seed(FUZZ_SEED)


def generate_fuzzed_basic_type():
    type_choice = random.choice(
        ["int", "float", "str", "bool", "none", "bytes"]
    )
    if type_choice == "int":
        return random.randint(FUZZ_MIN_INT, FUZZ_MAX_INT)
    elif type_choice == "float":
        special_floats = [float("inf"), float("-inf"), float("nan"), 0.0, -0.0, 1e-5, -1e-5]
        if random.random() < 0.25:  # Increased chance of special floats
            return random.choice(special_floats)
        return random.uniform(-1e5, 1e5)  # Smaller float range
    elif type_choice == "str":
        length = random.randint(0, FUZZ_MAX_STRING_LENGTH)
        # Include more diverse characters for fuzzing strings
        char_options = string.printable + "你好世界äöüßçñéàèìòù"
        return "".join(random.choice(char_options) for _ in range(length))
    elif type_choice == "bool":
        return random.choice([True, False])
    elif type_choice == "bytes":
        length = random.randint(0, FUZZ_MAX_BYTES_LENGTH)
        return bytes(random.getrandbits(8) for _ in range(length))
    else:  # none
        return None


def generate_fuzzed_object(current_depth=0, memoized_objects=None):
    if memoized_objects is None: memoized_objects = []

    # Chance to reuse an already created object (for memoization fuzzing)
    if current_depth > 0 and memoized_objects and random.random() < 0.1:
        return random.choice(memoized_objects)

    if current_depth >= FUZZ_MAX_DEPTH:
        obj = generate_fuzzed_basic_type()
        if isinstance(obj, (list, tuple, set, dict, str, bytes)) and len(memoized_objects) < 10:  # Limit memo size
            if obj not in memoized_objects: memoized_objects.append(obj)  # Only memoize if not already there
        return obj

    # Bias towards basic types at deeper levels or by chance
    if current_depth > 0 and random.random() < 0.3 * current_depth:  # Higher chance for basic with depth
        type_choice = 'basic'
    else:
        type_choice = random.choice(["basic", "list", "tuple", "dict", "set"])

    obj_result = None
    if type_choice == "basic":
        obj_result = generate_fuzzed_basic_type()
    elif type_choice == "list":
        size = random.randint(0, FUZZ_MAX_COLLECTION_SIZE)
        obj_result = [generate_fuzzed_object(current_depth + 1, memoized_objects) for _ in range(size)]
    elif type_choice == "tuple":
        size = random.randint(0, FUZZ_MAX_COLLECTION_SIZE)
        obj_result = tuple(
            generate_fuzzed_object(current_depth + 1, memoized_objects) for _ in range(size)
        )
    elif type_choice == "dict":
        size = random.randint(0, FUZZ_MAX_COLLECTION_SIZE)
        d = {}
        for _ in range(size):
            # Keys must be hashable and preferably simple for broader compatibility
            key_type_choice = random.choice(
                ["str_key", "int_key", "bytes_key", "bool_key", "none_key", "simple_tuple_key"])
            key = None
            if key_type_choice == "str_key":
                key_len = random.randint(1, 8)  # Shorter keys
                key = "".join(random.choice(string.ascii_letters) for _ in range(key_len))
            elif key_type_choice == "int_key":
                key = random.randint(FUZZ_MIN_INT // 100, FUZZ_MAX_INT // 100)  # Smaller range for keys
            elif key_type_choice == "bytes_key":
                key_len = random.randint(1, 8)
                key = bytes(random.getrandbits(8) for _ in range(key_len))
            elif key_type_choice == "bool_key":
                key = random.choice([True, False])
            elif key_type_choice == "none_key":
                key = None
            elif key_type_choice == "simple_tuple_key":  # Tuple of basics as key
                key = (generate_fuzzed_basic_type(), generate_fuzzed_basic_type())
                try:
                    hash(key)  # Ensure key is hashable
                except TypeError:
                    key = f"fallback_key_{random.randint(0, 100)}"  # Fallback if generated tuple isn't hashable

            d[key] = generate_fuzzed_object(current_depth + 1, memoized_objects)
        obj_result = d
    elif type_choice == "set":
        size = random.randint(0, FUZZ_MAX_COLLECTION_SIZE)
        s = set()
        for _ in range(size):
            elem = generate_fuzzed_object(current_depth + 1, memoized_objects)
            try:  # Elements of a set must be hashable
                hash(elem)
                s.add(elem)
            except TypeError:  # If unhashable (e.g. list), add a hashable fallback
                s.add(generate_fuzzed_basic_type())
        obj_result = s

    # Add to memoized objects if it's a container or a non-basic type that's potentially sharable
    if isinstance(obj_result, (list, tuple, set, dict, str, bytes)) and len(memoized_objects) < 10:
        if obj_result not in memoized_objects: memoized_objects.append(obj_result)
    return obj_result


test_cases_fuzzing = []
fuzz_memo_objects = []  # Shared memo list for a single fuzzing session
for i in range(FUZZ_NUM_TEST_CASES):
    fuzzed_data = None
    fuzz_memo_objects.clear()  # Reset for each top-level fuzzed object for some independence
    try:
        fuzzed_data = generate_fuzzed_object(memoized_objects=fuzz_memo_objects)
        desc_str = f"Fuzzed data (seed={FUZZ_SEED}, depth<={FUZZ_MAX_DEPTH}, size<={FUZZ_MAX_COLLECTION_SIZE})"
        try:
            # Try to get a short repr, but handle potential errors during repr itself for very complex objects
            data_repr_short = get_limited_repr(fuzzed_data, 70)  # Use helper
            desc_str = f"{desc_str}: {data_repr_short}"
        except Exception:  # Fallback if repr fails
            desc_str = f"{desc_str}: <data repr error>"
        test_cases_fuzzing.append(
            {"id": f"fuzz_{i + 1}", "data": fuzzed_data, "desc": desc_str}
        )
    except RecursionError:
        print(f"RecursionError generating fuzzed test case {i + 1}. Skipping.")
        test_cases_fuzzing.append(
            {
                "id": f"fuzz_{i + 1}_generation_error", "data": None,
                "desc": f"RecursionError during generation of fuzzed object (seed={FUZZ_SEED})",
            }
        )
    except Exception as e:
        print(f"Error generating fuzzed test case {i + 1}: {type(e).__name__} - {e}. Skipping this one.")
        test_cases_fuzzing.append(
            {
                "id": f"fuzz_{i + 1}_generation_error", "data": None,
                "desc": f"Error during generation of fuzzed object (seed={FUZZ_SEED}): {type(e).__name__} - {e}",
            }
        )


# --- Test Execution ---
def run_tests(test_cases, category_name, protocols_to_test):
    """
    Runs pickle-hash tests for a list of test cases.
    Returns two dictionaries: one for summary results, one for detailed results with input repr.
    """
    category_summary_results = {}
    category_detailed_results = {}
    print(f"\n\n{'=' * 10} CATEGORY: {category_name} {'=' * 10}")

    for case in test_cases:
        case_id = case["id"]
        data = case["data"]
        desc = case["desc"]
        input_repr_str = get_limited_repr(data)  # Get representation of input data

        if "generation_error" in case_id and data is None:
            print(f"\n--- SKIPPING Case ID: {case_id} (Data generation failed) ---")
            print(f"Description: {desc}")
            error_entry = {"description": desc, "hashes": {"ERROR": "Data generation failed"}}
            category_summary_results[case_id] = error_entry
            category_detailed_results[case_id] = {**error_entry, "input_repr": "<Data generation failed>"}
            continue

        summary_entry = {"description": desc, "hashes": {}}
        detailed_entry = {"description": desc, "input_repr": input_repr_str, "hashes": {}}

        print(f"\n--- Testing Case ID: {case_id} ---")
        print(f"Description: {desc}")
        # print(f"Input Data Repr (limited): {input_repr_str[:100]}{'...' if len(input_repr_str) > 100 else ''}")

        for protocol in protocols_to_test:
            proto_name_str = (
                str(protocol)
                if protocol is not None
                else f"Default({pickle.DEFAULT_PROTOCOL})"
            )
            proto_key_name = (  # Consistent key for JSON
                f"Protocol_{protocol}"
                if protocol is not None
                else f"Protocol_Default_{pickle.DEFAULT_PROTOCOL}"  # Make default proto explicit in key
            )

            current_hash = get_pickle_hash(
                data, protocol=protocol, note=f"{case_id} (Proto: {proto_name_str})"
            )
            summary_entry["hashes"][proto_key_name] = current_hash
            detailed_entry["hashes"][proto_key_name] = current_hash
            print(f"  {proto_key_name}: {current_hash}")

            # For sets, unstable custom objects, or fuzzed data, check for variation with default protocol
            # This is because hash randomization can affect iteration order of sets/dicts,
            # which can change pickle output if not handled carefully by custom types or pickle itself.
            is_potentially_unstable_type = "set" in desc.lower() or \
                                           "EXPECT POTENTIAL VARIATION" in desc.upper() or \
                                           "fuzz_" in case_id.lower()

            if is_potentially_unstable_type and protocol is None and not current_hash.startswith("ERROR:"):
                print(
                    f"    (Verifying potential instability for '{case_id}' with default "
                    f"protocol, 2 more runs):"
                )
                for i_run in range(2):  # Perform 2 extra runs
                    h_extra = get_pickle_hash(
                        data,
                        protocol=protocol,  # Still default protocol
                        note=f"{case_id} default proto, extra run {i_run + 2}",
                    )
                    print(f"      Run {i_run + 2}: {h_extra}")
                    if h_extra != current_hash and not h_extra.startswith("ERROR:"):
                        print(f"      WARNING: Hash mismatch on extra run for {case_id} with default protocol!")
                        # Store this mismatch indication if desired
                        detailed_entry["hashes"][f"{proto_key_name}_extra_run_{i_run + 2}_mismatch"] = h_extra

        category_summary_results[case_id] = summary_entry
        category_detailed_results[case_id] = detailed_entry

    return category_summary_results, category_detailed_results


# --- White-box Test Execution ---
def run_whitebox_analysis():
    """Run white-box testing analysis on the pickle module's source."""
    print("\n\n{'=' * 20} WHITE-BOX ANALYSIS {'=' * 20}")

    analyzer = PickleWhiteBoxAnalyzer()
    analysis_data = {
        "pickle_module_path": analyzer.pickle_module_path,
        "source_analysis_performed": False,
        "functions_found_ast": 0,
        "control_flow_structures": 0,
        "critical_functions_identified": {},
        "all_definitions_count": 0,
        "all_uses_count": 0,
        "def_use_pairs_count": 0,
        "coverage_report": None,
        "def_use_reports": None  # To store detailed def-use if needed
    }

    print("\n--- Analyzing Pickle Module Source ---")
    if analyzer.load_pickle_source():  # This also parses AST
        analyzer.analyze_source_structure()  # This does AST func defs, CFG, and Def-Use
        analysis_data["source_analysis_performed"] = True

        print(f"Pickle module path: {analyzer.pickle_module_path}")
        print(f"AST-based Function Definitions found: {len(analyzer.function_defs_ast)}")
        analysis_data["functions_found_ast"] = len(analyzer.function_defs_ast)

        print(f"Control flow structures (If, For, While): {len(analyzer.control_flow_paths)}")
        analysis_data["control_flow_structures"] = len(analyzer.control_flow_paths)

        critical_funcs = analyzer.get_critical_functions()
        analysis_data["critical_functions_identified"] = critical_funcs
        if critical_funcs:
            print("\nCritical functions (from AST) identified:")
            for func_name, func_info in critical_funcs.items():
                print(f"  - {func_name}: line {func_info['lineno']}, "
                      f"{len(func_info['args'])} args, "
                      f"{func_info['body_size']} AST body statements")

        # Report on Def-Use analysis
        defs_report = analyzer.get_definitions_report()
        uses_report = analyzer.get_uses_report()
        du_pairs_report = analyzer.get_def_use_pairs_report()

        analysis_data["all_definitions_count"] = sum(len(locs) for locs in defs_report.values())
        analysis_data["all_uses_count"] = sum(len(locs) for locs in uses_report.values())
        analysis_data["def_use_pairs_count"] = len(du_pairs_report)
        # Store the detailed reports if they are to be saved in JSON
        analysis_data["def_use_reports"] = {
            "definitions": {k: sorted(list(set(v))) for k, v in defs_report.items()},  # list of (line,scope)
            "uses": {k: sorted(list(set(v))) for k, v in uses_report.items()},  # list of (line,scope)
            "def_use_pairs": du_pairs_report  # Already sorted list of dicts
        }

        print(f"\n--- Def-Use Analysis (from AST) ---")
        print(f"  Total variable definition instances: {analysis_data['all_definitions_count']}")
        print(f"  Total variable use instances: {analysis_data['all_uses_count']}")
        print(f"  Total def-use pairs found: {analysis_data['def_use_pairs_count']}")
        if du_pairs_report:
            print("  Sample Def-Use Pairs (max 10 shown):")
            for pair in du_pairs_report[:10]:
                print(
                    f"    Var: {pair['variable']}, Def: L{pair['def_line']}({pair['def_scope']}) -> Use: L{pair['use_line']}({pair['use_scope']})")

    else:  # Python source not loaded (likely C implementation or error)
        print("Pickle module appears to be C-implemented or source not found/parsed.")
        print("AST-based white-box analysis (functions, def-use) cannot be performed.")
        if analyzer.c_pickle_functions:
            print("Known C pickle components (from _load_c_pickle_info):")
            for component, info in analyzer.c_pickle_functions.items():
                print(f"  - {component}: {info}")
        analysis_data["notes"] = "Pickle module likely C-implemented; AST analysis not performed."

    # Generate line/branch coverage report using `coverage` package
    print("\n--- Generating Line/Branch Coverage Report (using 'coverage' package) ---")
    # Ensure MyObjectStable is defined if used in coverage generation
    coverage_summary = analyzer.generate_coverage_report()
    analysis_data["coverage_report"] = coverage_summary
    if coverage_summary and 'error' not in coverage_summary and 'info' not in coverage_summary:
        for file, info in coverage_summary.items():
            print(f"\nCoverage for: {file} ({info.get('path', '')})")
            print(f"  Total Statements: {info.get('total_statements', 'N/A')}")
            print(f"  Executed Statements: {info.get('executed_statements', 'N/A')}")
            print(f"  Percent Covered: {info.get('percent_covered', 'N/A')}")
            if info.get('sample_missing_lines'):
                print(f"  Sample Missing Lines: {info['sample_missing_lines']}")
    elif coverage_summary and 'error' in coverage_summary:
        print(f"Coverage analysis error: {coverage_summary['error']}")
    elif coverage_summary and 'info' in coverage_summary:
        print(f"Coverage info: {coverage_summary['info']}")
    else:
        print("Coverage analysis did not produce data.")

    return analysis_data  # Return all collected whitebox data


def run_path_coverage_tests(protocols_to_test):
    """Run path-specific coverage tests designed to hit particular logic in pickle."""
    print("\n\n{'=' * 20} PATH COVERAGE TESTS {'=' * 20}")

    path_tester = PicklePathCoverageTester()
    path_test_cases = path_tester.get_test_cases()

    path_results_summary = {}  # For summary file
    path_results_detailed = {}  # For detailed file (includes input_repr if needed, though path tests focus on behavior)

    for case in path_test_cases:
        case_id = case["id"]
        data = case["data"]
        desc = case["desc"]
        expect_error = case.get("expect_error", False)
        input_repr_str = get_limited_repr(data)

        print(f"\n--- Path Test: {case_id} ---")
        print(f"Description: {desc}")

        current_case_summary_hashes = {}
        current_case_detailed_hashes = {}

        if "Generation failed" in desc and data is None:  # Handle generation failure
            result_val = f"ERROR: Test case generation failed - {desc.split('failed: ')[-1]}"
            for protocol in protocols_to_test:
                proto_key_name = f"Protocol_{protocol}" if protocol is not None else f"Protocol_Default_{pickle.DEFAULT_PROTOCOL}"
                current_case_summary_hashes[proto_key_name] = result_val
                current_case_detailed_hashes[proto_key_name] = result_val
        else:
            for protocol in protocols_to_test:
                proto_key_name = f"Protocol_{protocol}" if protocol is not None else f"Protocol_Default_{pickle.DEFAULT_PROTOCOL}"
                result_val = ""
                try:
                    if expect_error:
                        try:
                            # For expected errors, we try to pickle and expect an exception
                            pickle.dumps(data, protocol=protocol)
                            result_val = f"UNEXPECTED_SUCCESS (Expected Error for {desc})"  # Should have failed
                        except Exception as e:
                            result_val = f"EXPECTED_ERROR: {type(e).__name__} - {str(e)[:100]}"  # Store type of error
                    else:
                        # For normal cases, get the hash
                        result_val = get_pickle_hash(data, protocol=protocol, note=f"Path test {case_id}")
                except Exception as e_outer:  # Catch errors during the test execution itself
                    result_val = f"ERROR_DURING_TEST_EXEC: {type(e_outer).__name__} - {str(e_outer)[:100]}"

                current_case_summary_hashes[proto_key_name] = result_val
                current_case_detailed_hashes[proto_key_name] = result_val
                print(f"  {proto_key_name}: {str(result_val)[:70]}...")  # Print truncated result

        path_results_summary[case_id] = {"description": desc, "hashes": current_case_summary_hashes}
        path_results_detailed[case_id] = {"description": desc, "input_repr": input_repr_str,
                                          "hashes": current_case_detailed_hashes}

    return path_results_summary, path_results_detailed


def generate_traceability_report(all_results_summary):
    """Generate traceability matrix report based on features and run test IDs."""
    print("\n\n{'=' * 20} TRACEABILITY MATRIX {'=' * 20}")

    matrix_builder = TraceabilityMatrix()  # Renamed variable for clarity
    # Pass all_results_summary which contains {category_name: {test_id: ...}}
    report = matrix_builder.generate_matrix_report(all_results_summary)

    print("\n--- Feature Coverage Summary ---")
    for category, features in sorted(report['feature_coverage'].items()):
        print(f"\nCategory: {category}:")
        for feature, coverage_info in sorted(features.items()):
            note = f" ({coverage_info['notes']})" if coverage_info['notes'] else ""
            print(f"  - Feature: {feature}: {coverage_info['status']}{note}")
            if coverage_info.get('covering_test_ids'):
                print(
                    f"    Covered by: {coverage_info['covering_test_ids'][:5]}{'...' if len(coverage_info['covering_test_ids']) > 5 else ''}")

    if report['uncovered_features']:
        print("\n--- Potentially Uncovered or Partially Covered Features (requiring review) ---")
        for feature_path in sorted(report['uncovered_features']):
            print(f"  - {feature_path}")
    else:
        print("\n--- All defined features appear to have associated test coverage (review status for details). ---")

    # print("\n--- Test Case to Feature Mapping (Sample) ---")
    # sample_count = 0
    # for test_id, mapped_features in report['test_to_feature_mapping'].items():
    #     if sample_count < 5:
    #         print(f"  Test '{test_id}' maps to: {mapped_features}")
    #         sample_count +=1
    #     else:
    #         print(f"  ... and {len(report['test_to_feature_mapping']) - sample_count} more test mappings.")
    #         break

    return report  # This report will be saved to the whitebox JSON


if __name__ == "__main__":
    # Existing test data categories
    all_test_data_categories = {
        "Basic Types": test_cases_basic,
        "Container Types": test_cases_containers,
        "Complex Structures": test_cases_complex,
        "Recursive Structures": test_cases_recursive,
        "Custom Objects": test_cases_custom_objects,
        "Advanced Pickling Control": test_cases_advanced_pickling,
        "Fuzzing": test_cases_fuzzing,
    }

    # Determine protocols to test with
    protocols_to_test_core = [None]  # None means default protocol
    # Add specific protocol numbers, ensuring HIGHEST_PROTOCOL is included
    # Also add 0 for completeness if not default/highest.
    if 0 <= pickle.HIGHEST_PROTOCOL: protocols_to_test_core.append(0)

    # Add protocols 1 through HIGHEST_PROTOCOL
    for p_val in range(1, pickle.HIGHEST_PROTOCOL + 1):
        protocols_to_test_core.append(p_val)

    # Remove duplicates and sort, keeping None at the beginning.
    seen_protocols = set()
    final_protocols_to_test = []
    if None in protocols_to_test_core:
        final_protocols_to_test.append(None)
        seen_protocols.add("Default")  # Mark default as seen conceptually

    numeric_protocols = sorted(list(set(p for p in protocols_to_test_core if isinstance(p, int))))
    final_protocols_to_test.extend(numeric_protocols)

    print(f"Python Version: {sys.version.splitlines()[0]}")
    print(f"Platform: {sys.platform}")
    print(f"Default Pickle Protocol: {pickle.DEFAULT_PROTOCOL}")
    print(f"Highest Pickle Protocol: {pickle.HIGHEST_PROTOCOL}")
    print(
        "Testing with Protocols: "
        f"{[p if p is not None else f'Default({pickle.DEFAULT_PROTOCOL})' for p in final_protocols_to_test]}"
    )

    environment_summary_results = {}  # For hashes only
    environment_detailed_results = {}  # For inputs and hashes

    # Run white-box analysis first to gather static info
    # This function now returns a dictionary of its findings.
    whitebox_analysis_results_dict = run_whitebox_analysis()

    # Run original black-box tests (stability/hash tests)
    for category_name, test_data_list in all_test_data_categories.items():
        summary_res, detailed_res = run_tests(
            test_data_list, category_name, final_protocols_to_test
        )
        environment_summary_results[category_name] = summary_res
        environment_detailed_results[category_name] = detailed_res

    # Run path coverage tests (also generates hashes, but for specific scenarios)
    # This now returns two dicts: summary and detailed
    path_summary_res, path_detailed_res = run_path_coverage_tests(final_protocols_to_test)
    environment_summary_results["Path Coverage Tests"] = path_summary_res
    environment_detailed_results["Path Coverage Tests"] = path_detailed_res

    # Generate traceability matrix using the summary results (which have all test IDs)
    traceability_report_dict = generate_traceability_report(environment_summary_results)

    print("\n\n--- All Test Data Collected (for this environment) ---")

    py_version_slug = f"py{sys.version_info.major}{sys.version_info.minor}"
    platform_slug = sys.platform.replace("-", "_").replace(" ", "_").lower()  # Sanitize platform string

    # --- Output File Generation ---

    # File 1: Hashes only (original summary format)
    output_filename_summary = f"pickle_hashes_{py_version_slug}_{platform_slug}_seed{FUZZ_SEED}.json"
    try:
        with open(output_filename_summary, "w", encoding="utf-8") as f:
            json.dump(environment_summary_results, f, indent=2, ensure_ascii=False)
        print(f"\nSummary results (hashes only) saved to {output_filename_summary}")
    except Exception as e:
        print(f"\nError saving summary results to {output_filename_summary}: {e}")

    # File 2: Inputs and Hashes (detailed format)
    output_filename_detailed = f"pickle_inputs_and_hashes_{py_version_slug}_{platform_slug}_seed{FUZZ_SEED}.json"
    try:
        with open(output_filename_detailed, "w", encoding="utf-8") as f:
            json.dump(environment_detailed_results, f, indent=2, ensure_ascii=False)
        print(f"Detailed results (inputs and hashes) saved to {output_filename_detailed}")
    except Exception as e:
        print(f"\nError saving detailed results to {output_filename_detailed}: {e}")

    # File 3: White-box analysis results (consolidated)
    output_filename_whitebox = f"pickle_whitebox_analysis_{py_version_slug}_{platform_slug}.json"
    # Combine whitebox_analysis_results_dict with traceability_report_dict
    # The whitebox_analysis_results_dict already contains sections for AST analysis and coverage.
    # We just need to add the traceability matrix to it.

    # Structure for whitebox JSON:
    # whitebox_analysis_results_dict already has:
    #   pickle_module_path, source_analysis_performed, functions_found_ast, etc.
    #   def_use_reports, coverage_report
    # Add traceability matrix to this dict:
    whitebox_analysis_results_dict["traceability_matrix"] = traceability_report_dict
    # Path coverage test results are more like functional tests, already in summary/detailed.
    # If specific path coverage metrics (not just pass/fail/hash) were derived, they could go here.
    # For now, path test outcomes are in the main result files.

    try:
        with open(output_filename_whitebox, "w", encoding="utf-8") as f:
            # Custom encoder for sets or other non-serializable types if they appear
            class CustomEncoder(json.JSONEncoder):
                def default(self, o):
                    if isinstance(o, set):
                        return sorted(list(o))  # Serialize sets as sorted lists
                    if isinstance(o, defaultdict):
                        return dict(o)
                    return super().default(o)


            json.dump(whitebox_analysis_results_dict, f, indent=2, ensure_ascii=False, cls=CustomEncoder)
        print(f"White-box analysis results saved to {output_filename_whitebox}")
    except Exception as e:
        print(f"\nError saving white-box results to {output_filename_whitebox}: {e}")
        traceback.print_exc()

    print("\n--- Next Steps (Reminder) ---")
    print("1. Run this script in VARIOUS environments (different OS, Python versions).")
    print("2. Collect and compare the JSON output files from each environment.")
    print("   - Use the 'pickle_hashes_...' files for direct hash comparison (stability).")
    print("   - Use the 'pickle_inputs_and_hashes_...' files for more context on what was pickled.")
    print(
        "   - Use the 'pickle_whitebox_analysis_...' files for code structure, def-use insights, coverage, and traceability.")
    print("3. Any differing hashes for the *same test case ID and same protocol* "
          "across environments indicate an instability.")
    print("   (With the fixed seed, fuzzed cases are also directly comparable).")
    print("4. The white-box analysis (AST, Def-Use, Coverage) provides insight into pickle's internal structure "
          "and helps assess test thoroughness against the code itself.")
    print(
        "5. Review the traceability matrix to ensure pickle features are adequately tested by the defined test cases.")
    print("6. Consider additional white-box tests based on uncovered code paths (from coverage) or unlinked defs/uses.")
