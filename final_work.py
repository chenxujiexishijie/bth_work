# test_pickle_stability_enhanced.py

import pickle
import hashlib
import sys
import math  # For inf, nan
import random
import string
from decimal import Decimal  # For precise float-like behavior if needed
import json  # For output


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
            f"{protocol if protocol is not None else 'Default (' + str(pickle.DEFAULT_PROTOCOL) + ')'}: {e}"
        )
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
        "data": {1: "int_key", "str_key": 2, (1, 2): "tuple_key"},
        "desc": "Dictionary with mixed key types",
    },
    {
        "id": "dict_ordered_keys_insertion",
        "data": {"z": 1, "y": 2, "x": 3, "a": 4, "b": 5},
        "desc": "Dict with specific insertion order (Python 3.7+ preserves it)",
    },
    {
        "id": "set_simple",
        "data": {1, "two", 3.0, None, True},
        "desc": "Simple set (EXPECT POTENTIAL VARIATION for protocols < 4 or across Pythons)",
    },
    {"id": "set_empty", "data": set(), "desc": "Empty set"},
    {
        "id": "set_single_element",
        "data": {"hello"},
        "desc": "Set with a single element",
    },
    {
        "id": "set_all_same_type",
        "data": {10, 20, 30, 1, 2, 3},
        "desc": "Set with elements of same type",
    },
    {
        "id": "set_with_special_floats",
        "data": {0.0, -0.0, float("inf"), float("-inf"), float("nan")},
        "desc": "Set containing special floats (NaN makes it tricky)",
    },
    {
        "id": "set_from_list_with_duplicates",
        "data": set([1, 2, 2, 3, 3, 3, 4]),
        "desc": "Set created from list with duplicates",
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
for _ in range(10):
    new_level = [temp_list_for_deep_list]
    temp_list_for_deep_list.append(new_level)
    temp_list_for_deep_list = new_level
temp_list_for_deep_list.append("end_orig_deep")

simple_deep_item = "end_simple_deep"
for _ in range(30):
    simple_deep_item = [simple_deep_item]
simple_deep_list_data = simple_deep_item

complex_shared_obj = ["shared_part"]
structure_with_shared = [
    complex_shared_obj,
    complex_shared_obj,
    {"key": complex_shared_obj},
]

test_cases_complex = [
    {
        "id": "list_nested_basic",
        "data": [1, [2, 3], [4, [5, "six"]]],
        "desc": "Basic nested list",
    },
    {
        "id": "list_deeply_nested_orig_logic",
        "data": deep_list_orig_logic,
        "desc": "Deeply nested list using original recursive logic (depth approx 10)",
    },
    {
        "id": "list_deeply_nested_simple",
        "data": simple_deep_list_data,
        "desc": "Deeply nested list (non-recursive def, depth 30)",
    },
    {
        "id": "dict_nested_basic",
        "data": {"k1": "v1", "k2": [1, 2, {"sk": "sv"}], "k3": {}},
        "desc": "Basic nested dictionary",
    },
    {
        "id": "list_of_dicts_varied",
        "data": [
            {"id": 1, "val": "a", "active": True},
            {"id": 2, "val": None, "tags": ["x", "y"]},
        ],
        "desc": "List of dictionaries with varied content",
    },
    {
        "id": "dict_with_list_of_sets",
        "data": {"data_sets": [{1, 2}, {2, 3, 4}, frozenset({5, 6})]},
        "desc": "Dict containing list of sets/frozensets",
    },
    {
        "id": "structure_with_shared_obj",
        "data": structure_with_shared,
        "desc": "Structure with internally shared object reference",
    },
    {
        "id": "list_containing_all_special_floats",
        "data": [float("inf"), float("-inf"), float("nan"), -0.0, 0.0],
        "desc": "List with all special floats in specific order",
    },
    {
        "id": "set_containing_tuples_with_nans",
        "data": {(1, float("nan")), (2, 0.0), (1, float("nan"))},
        "desc": "Set of tuples where tuples contain NaN (complex comparison)",
    },
]

# 4. Recursive Structures
simple_recursive_list_data = []
simple_recursive_list_data.append(simple_recursive_list_data)

simple_recursive_dict_data = {}
simple_recursive_dict_data["self"] = simple_recursive_dict_data

recursive_list_A_data = ["A"]
recursive_list_B_data = ["B"]
recursive_list_A_data.append(recursive_list_B_data)
recursive_list_B_data.append(recursive_list_A_data)

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
        "id": "list_AB_recursion",
        "data": recursive_list_A_data,
        "desc": "Mutually recursive lists A <-> B",
    },
    {
        "id": "obj_indirect_recursion_complex_shared_part",
        "data": structure_with_shared[2],
        "desc": "Shared object part of a complex structure (re-test)",
    },
]


# 5. Custom Objects (Extended to show __getstate__/__setstate__ impact)
class MyObjectStable:
    def __init__(self, name, value, items_set=None):
        self.name = name
        self.value = value
        self.items_set = items_set if items_set is not None else set()

    def __getstate__(self):
        state = self.__dict__.copy()
        try:
            state["items_set"] = sorted(list(self.items_set))
        except TypeError:
            state["items_set"] = sorted(list(map(repr, self.items_set)))
        return state

    def __setstate__(self, state):
        state["items_set"] = set(state["items_set"])
        self.__dict__.update(state)

    def __str__(self):
        items_repr = (
            sorted(list(self.items_set))
            if self.items_set
            else set()
        )
        return (
            f"MyObjectStable(name='{self.name}', value={self.value}, "
            f"items_set={items_repr})"
        )

    def __repr__(self):  # Added for better representation in output
        return f"MyObjectStable({self.name!r}, {self.value!r}, {self.items_set!r})"


class MyObjectUnstable:
    def __init__(self, name, value, items_set=None):
        self.name = name
        self.value = value
        self.items_set = items_set if items_set is not None else set()

    def __str__(self):
        return (
            f"MyObjectUnstable(name='{self.name}', value={self.value}, "
            f"items_set={self.items_set})"
        )

    def __repr__(self):  # Added for better representation in output
        return f"MyObjectUnstable({self.name!r}, {self.value!r}, {self.items_set!r})"


custom_obj_stable = MyObjectStable("stable", 100, {"gamma", "alpha", "beta"})
custom_obj_unstable = MyObjectUnstable(
    "unstable", 200, {"zeta", "epsilon", "delta"}
)

MyObjectDefault = type(
    "MyObjectDefault",
    (object,),
    {
        "__init__": lambda self, x, y_set: setattr(self, "attr1", x)
                                           or setattr(self, "attr2_set", y_set),
        "attr1": "default_val1",
        "attr2_set": {"default_set_item"},
        "__str__": lambda self: (
            f"MyObjectDefault(attr1='{self.attr1}', attr2_set={self.attr2_set})"
        ),
        "__repr__": lambda self: (  # Added for better representation
            f"MyObjectDefault({getattr(self, 'attr1', 'N/A')!r}, {getattr(self, 'attr2_set', 'N/A')!r})"
        )
    },
)
custom_obj_default_instance = MyObjectDefault("val1_instance", {"data_set_instance"})

test_cases_custom_objects = [
    {
        "id": "custom_obj_stable_getstate",
        "data": custom_obj_stable,
        "desc": str(custom_obj_stable),
    },
    {
        "id": "custom_obj_unstable_no_getstate_with_set_attr",
        "data": custom_obj_unstable,
        "desc": str(custom_obj_unstable) + " (EXPECT POTENTIAL VARIATION)",
    },
    {
        "id": "custom_obj_default_with_set_attr",
        "data": custom_obj_default_instance,
        "desc": f"{str(custom_obj_default_instance)} (EXPECT POTENTIAL VARIATION for set)",
    },
]


# 6. Advanced Pickling Control / White-box Inspired
class ReduciblePoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __reduce__(self):
        return (self.__class__, (self.x, self.y))

    def __str__(self):
        return f"ReduciblePoint({self.x}, {self.y})"

    def __repr__(self):  # Added
        return f"ReduciblePoint({self.x!r}, {self.y!r})"


class ReduciblePointEx(ReduciblePoint):
    def __init__(self, x, y, z=0):
        super().__init__(x, y)
        self.z = z
        self.extra_state = {"info": "from reduce_ex", "version": 1}

    def __reduce_ex__(self, protocol):
        return (
            self.__class__,
            (self.x, self.y, self.z),
            self.extra_state,
            None,
            None,
        )

    def __setstate__(self, state):
        self.extra_state = state

    def __str__(self):
        return (
            f"ReduciblePointEx({self.x}, {self.y}, {self.z}, "
            f"extra={self.extra_state})"
        )

    def __repr__(self):  # Added
        return f"ReduciblePointEx({self.x!r}, {self.y!r}, {self.z!r}, extra_state={self.extra_state!r})"


class MemoTricky:
    _instance_cache = {}

    def __new__(cls, name):
        if name not in cls._instance_cache:
            instance = super().__new__(cls)
            instance.name = name
            instance.data = []
            cls._instance_cache[name] = instance
        return cls._instance_cache[name]

    def __init__(self, name):
        if not hasattr(self, 'initialized_by_init'):
            self.initialized_by_init = True

    def add_data(self, item):
        self.data.append(item)

    def __str__(self):
        return f"MemoTricky(name='{self.name}', data={self.data})"

    def __repr__(self):  # Added
        return f"MemoTricky({self.name!r})"  # Keep repr simple to avoid recursion issues with data

    def __getstate__(self):
        return {"name": self.name, "data": self.data}

    def __setstate__(self, state):
        self.name = state["name"]
        self.data = state["data"]
        self.initialized_by_init = True


MemoTricky._instance_cache.clear()
mt_shared_A = MemoTricky("shared_A")
mt_shared_A.add_data(1)
mt_shared_A.add_data(2)
mt_unique_B = MemoTricky("unique_B")
mt_unique_B.add_data(100)
memo_test_list = [mt_shared_A, mt_unique_B, mt_shared_A, MemoTricky("another_A")]

test_cases_advanced_pickling = [
    {
        "id": "reducible_point",
        "data": ReduciblePoint(10, 20),
        "desc": str(ReduciblePoint(10, 20)),
    },
    {
        "id": "reducible_point_ex",
        "data": ReduciblePointEx(30, 40, 50),
        "desc": str(ReduciblePointEx(30, 40, 50)),
    },
    {
        "id": "memo_tricky_list_with_shared",
        "data": memo_test_list,
        "desc": f"List with shared MemoTricky instances: {[str(x) for x in memo_test_list]}",
    },
]
MemoTricky._instance_cache.clear()

# 7. Fuzzing Test Cases
FUZZ_NUM_TEST_CASES = 25
FUZZ_MAX_DEPTH = 3
FUZZ_MAX_COLLECTION_SIZE = 4
FUZZ_MAX_STRING_LENGTH = 50
FUZZ_MAX_BYTES_LENGTH = 50
FUZZ_MAX_INT = 2 ** 30 - 1
FUZZ_MIN_INT = -(2 ** 30)
FUZZ_SEED = 42
random.seed(FUZZ_SEED)


def generate_fuzzed_basic_type():
    type_choice = random.choice(
        ["int", "float", "str", "bool", "none", "bytes"]
    )
    if type_choice == "int":
        return random.randint(FUZZ_MIN_INT, FUZZ_MAX_INT)
    elif type_choice == "float":
        special_floats = [float("inf"), float("-inf"), float("nan"), 0.0, -0.0]
        if random.random() < 0.2:
            return random.choice(special_floats)
        return random.uniform(-1e10, 1e10)
    elif type_choice == "str":
        length = random.randint(0, FUZZ_MAX_STRING_LENGTH)
        return "".join(random.choice(string.printable) for _ in range(length))
    elif type_choice == "bool":
        return random.choice([True, False])
    elif type_choice == "bytes":
        length = random.randint(0, FUZZ_MAX_BYTES_LENGTH)
        return bytes(random.getrandbits(8) for _ in range(length))
    else:
        return None


def generate_fuzzed_object(current_depth=0):
    if current_depth >= FUZZ_MAX_DEPTH:
        return generate_fuzzed_basic_type()
    if current_depth > 0 and random.random() < 0.3 * current_depth:
        type_choice = 'basic'
    else:
        type_choice = random.choice(["basic", "list", "tuple", "dict", "set"])
    if type_choice == "basic":
        return generate_fuzzed_basic_type()
    elif type_choice == "list":
        size = random.randint(0, FUZZ_MAX_COLLECTION_SIZE)
        return [generate_fuzzed_object(current_depth + 1) for _ in range(size)]
    elif type_choice == "tuple":
        size = random.randint(0, FUZZ_MAX_COLLECTION_SIZE)
        return tuple(
            generate_fuzzed_object(current_depth + 1) for _ in range(size)
        )
    elif type_choice == "dict":
        size = random.randint(0, FUZZ_MAX_COLLECTION_SIZE)
        d = {}
        for _ in range(size):
            key_type_choice = random.choice(["str_key", "int_key", "bytes_key", "bool_key", "none_key"])
            key = None
            if key_type_choice == "str_key":
                key_len = random.randint(1, 10)
                key = "".join(random.choice(string.ascii_letters) for _ in range(key_len))
            elif key_type_choice == "int_key":
                key = random.randint(FUZZ_MIN_INT // 1000, FUZZ_MAX_INT // 1000)
            elif key_type_choice == "bytes_key":
                key_len = random.randint(1, 10)
                key = bytes(random.getrandbits(8) for _ in range(key_len))
            elif key_type_choice == "bool_key":
                key = random.choice([True, False])
            elif key_type_choice == "none_key":
                key = None
            d[key] = generate_fuzzed_object(current_depth + 1)
        return d
    elif type_choice == "set":
        size = random.randint(0, FUZZ_MAX_COLLECTION_SIZE)
        s = set()
        for _ in range(size):
            elem = generate_fuzzed_object(current_depth + 1)
            try:
                hash(elem)
                s.add(elem)
            except TypeError:
                s.add(generate_fuzzed_basic_type())  # Fallback
        return s
    return None


test_cases_fuzzing = []
for i in range(FUZZ_NUM_TEST_CASES):
    fuzzed_data = None
    try:
        fuzzed_data = generate_fuzzed_object()
        desc_str = f"Fuzzed data (seed={FUZZ_SEED}, depth<={FUZZ_MAX_DEPTH}, size<={FUZZ_MAX_COLLECTION_SIZE})"
        try:
            data_repr_short = repr(fuzzed_data)[:70]
            if len(repr(fuzzed_data)) > 70:
                data_repr_short += "..."
            desc_str = f"{desc_str}: {data_repr_short}"
        except Exception:
            pass
        test_cases_fuzzing.append(
            {"id": f"fuzz_{i + 1}", "data": fuzzed_data, "desc": desc_str}
        )
    except Exception as e:
        print(f"Error generating fuzzed test case {i + 1}: {e}. Skipping this one.")
        test_cases_fuzzing.append(
            {
                "id": f"fuzz_{i + 1}_generation_error",
                "data": None,
                "desc": f"Error during generation of fuzzed object (seed={FUZZ_SEED}): {e}",
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
        # Uncomment if a preview of the limited_repr is desired during console output
        # print(f"Input Data Repr (limited): {input_repr_str[:100]}{'...' if len(input_repr_str) > 100 else ''}")

        for protocol in protocols_to_test:
            proto_name_str = (
                str(protocol)
                if protocol is not None
                else f"Default({pickle.DEFAULT_PROTOCOL})"
            )
            proto_key_name = (
                f"Protocol_{protocol}"
                if protocol is not None
                else f"Default_Protocol_{pickle.DEFAULT_PROTOCOL}"
            )

            current_hash = get_pickle_hash(
                data, protocol=protocol, note=f"{case_id} (Proto: {proto_name_str})"
            )
            summary_entry["hashes"][proto_key_name] = current_hash
            detailed_entry["hashes"][proto_key_name] = current_hash
            print(f"  {proto_key_name}: {current_hash}")

            if (("EXPECT POTENTIAL VARIATION" in desc.upper() or "FUZZ_" in case_id.upper())
                    and protocol is None and not current_hash.startswith("ERROR:")):
                print(
                    f"    (Verifying potential instability for '{case_id}' with default "
                    f"protocol, 2 more runs):"
                )
                for i_run in range(2):
                    h = get_pickle_hash(
                        data,
                        protocol=protocol,
                        note=f"{case_id} default proto, extra run {i_run + 2}",
                    )
                    print(f"      Run {i_run + 2}: {h}")
                    if h != current_hash and not h.startswith("ERROR:"):
                        print(f"      WARNING: Hash mismatch on extra run for {case_id}!")

        category_summary_results[case_id] = summary_entry
        category_detailed_results[case_id] = detailed_entry

    return category_summary_results, category_detailed_results


if __name__ == "__main__":
    all_test_data_categories = {
        "Basic Types": test_cases_basic,
        "Container Types": test_cases_containers,
        "Complex Structures": test_cases_complex,
        "Recursive Structures": test_cases_recursive,
        "Custom Objects": test_cases_custom_objects,
        "Advanced Pickling Control": test_cases_advanced_pickling,
        "Fuzzing": test_cases_fuzzing,
    }

    protocols_to_test_core = [None, pickle.HIGHEST_PROTOCOL]
    for p_val in [0, 1, 2, 3, 4]:
        if p_val <= pickle.HIGHEST_PROTOCOL and p_val not in protocols_to_test_core:
            protocols_to_test_core.append(p_val)
    if sys.version_info >= (3, 8) and 5 <= pickle.HIGHEST_PROTOCOL:
        if 5 not in protocols_to_test_core:
            protocols_to_test_core.append(5)

    processed_protocols = []
    if None not in processed_protocols:  # Conceptually, 'None' means default
        processed_protocols.append(None)

    numeric_protocols = sorted(
        list(set(p for p in protocols_to_test_core if isinstance(p, int)))
    )

    final_protocols_to_test = []
    if None in processed_protocols:
        final_protocols_to_test.append(None)
    final_protocols_to_test.extend(numeric_protocols)

    print(f"Python Version: {sys.version.splitlines()[0]}")
    print(f"Platform: {sys.platform}")
    print(f"Default Pickle Protocol: {pickle.DEFAULT_PROTOCOL}")
    print(f"Highest Pickle Protocol: {pickle.HIGHEST_PROTOCOL}")
    print(
        "Testing with Protocols: "
        f"{[p if p is not None else f'Default({pickle.DEFAULT_PROTOCOL})' for p in final_protocols_to_test]}"
    )

    environment_summary_results = {}
    environment_detailed_results = {}  # For the new file with input repr

    for category_name, test_data_list in all_test_data_categories.items():
        summary_res, detailed_res = run_tests(
            test_data_list, category_name, final_protocols_to_test
        )
        environment_summary_results[category_name] = summary_res
        environment_detailed_results[category_name] = detailed_res

    print("\n\n--- All Test Data Collected (for this environment) ---")

    py_version_slug = f"py{sys.version_info.major}{sys.version_info.minor}"
    platform_slug = sys.platform.replace("-", "_").lower()

    # Output file 1: Hashes only (original format)
    output_filename_summary = f"pickle_hashes_{py_version_slug}_{platform_slug}_seed{FUZZ_SEED}.json"
    try:
        with open(output_filename_summary, "w", encoding="utf-8") as f:
            json.dump(environment_summary_results, f, indent=2, ensure_ascii=False)
        print(f"\nSummary results (hashes only) saved to {output_filename_summary}")
    except Exception as e:
        print(f"\nError saving summary results to {output_filename_summary}: {e}")

    # Output file 2: Inputs and Hashes (new format)
    output_filename_detailed = f"pickle_inputs_and_hashes_{py_version_slug}_{platform_slug}_seed{FUZZ_SEED}.json"
    try:
        with open(output_filename_detailed, "w", encoding="utf-8") as f:
            json.dump(environment_detailed_results, f, indent=2, ensure_ascii=False)
        print(f"Detailed results (inputs and hashes) saved to {output_filename_detailed}")
    except Exception as e:
        print(f"\nError saving detailed results to {output_filename_detailed}: {e}")

    print("\n--- Next Steps (Reminder) ---")
    print("1. Run this script in VARIOUS environments (different OS, Python versions).")
    print("2. Collect and compare the JSON output files from each environment.")
    print("   - Use the 'pickle_hashes_...' files for direct hash comparison.")
    print("   - Use the 'pickle_inputs_and_hashes_...' files for more context on what was pickled.")
    print("3. Any differing hashes for the *same test case ID and same protocol* "
          "across environments indicate an instability.")
    print("   (With the fixed seed, fuzzed cases are now also directly comparable).")
    print("4. This script provides a foundation. Consider deeper white-box analysis "
          "of pickle's C code if specific instabilities are found and need diagnosis.")
    print("5. Review test coverage (e.g., using a traceability matrix against pickle features).")
