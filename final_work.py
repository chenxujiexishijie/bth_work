# test_pickle_stability_enhanced.py

import pickle
import hashlib
import sys
import math # For inf, nan
import random
import string
from decimal import Decimal # For precise float-like behavior if needed

# --- Helper Function ---
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
        # For fuzzed objects, the object representation might be too long or complex
        # print(f"Failed object causing error: {str(data_object)[:200]}")
        return f"ERROR: {error_message}"


# --- Test Cases ---

# 1. Basic Data Types (Extended)
test_cases_basic = [
    {"id": "int_simple", "data": 123, "desc": "A simple integer"},
    {"id": "int_zero", "data": 0, "desc": "Integer zero"},
    {"id": "int_neg", "data": -42, "desc": "A negative integer"},
    {"id": "int_small_neg", "data": -1, "desc": "Small negative integer"},
    {
        "id": "int_large_pos",
        "data": 2**100,
        "desc": "A large positive integer (Python arbitrary precision)",
    },
    {"id": "int_large_neg", "data": -(2**100), "desc": "A large negative integer"},
    {
        "id": "int_boundary_32bit_pos",
        "data": 2**31 - 1,
        "desc": "Max positive 32-bit signed int",
    },
    {
        "id": "int_boundary_32bit_plus_one",
        "data": 2**31,
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
# Original deep_list logic, which creates a complex recursive structure
deep_list_orig_logic = [1]
temp_list_for_deep_list = deep_list_orig_logic
for _ in range(10):  # Reduced range from 50 to keep it manageable
    new_level = [temp_list_for_deep_list]
    temp_list_for_deep_list.append(new_level)
    temp_list_for_deep_list = new_level
temp_list_for_deep_list.append("end_orig_deep")

# Simpler deeply nested list (non-recursive definition)
simple_deep_item = "end_simple_deep"
for _ in range(30):  # Depth of nesting
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
# These are defined after list/dict types are available.
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
    { # This tests pickling a part of a structure that involves shared references.
      # It's not directly recursive itself but comes from a complex graph.
        "id": "obj_indirect_recursion_complex_shared_part",
        "data": structure_with_shared[2], # This is {"key": complex_shared_obj}
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
        # Ensure set is sorted for stable pickling output
        state = self.__dict__.copy()
        try:
            # Try sorting directly, works if elements are comparable
            state["items_set"] = sorted(list(self.items_set))
        except TypeError:
            # Fallback for non-comparable elements (e.g., mixed types, NaN)
            # Sort by representation for stability
            state["items_set"] = sorted(list(map(repr, self.items_set)))
        return state

    def __setstate__(self, state):
        # Convert sorted list back to set
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


custom_obj_stable = MyObjectStable("stable", 100, {"gamma", "alpha", "beta"})
custom_obj_unstable = MyObjectUnstable(
    "unstable", 200, {"zeta", "epsilon", "delta"}
)

# Dynamically created class for default behavior
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
# Classes using __reduce__ and __reduce_ex__ to control pickling.
# These methods are key to how pickle serializes custom objects.
# Testing them helps verify pickle's interaction with these protocols.

class ReduciblePoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __reduce__(self):
        # Returns: (callable_to_reconstruct, args_for_callable)
        return (self.__class__, (self.x, self.y))

    def __str__(self):
        return f"ReduciblePoint({self.x}, {self.y})"


class ReduciblePointEx(ReduciblePoint):
    def __init__(self, x, y, z=0):
        super().__init__(x, y)
        self.z = z
        self.extra_state = {"info": "from reduce_ex", "version": 1}

    def __reduce_ex__(self, protocol):
        # Returns: (callable, args, state, listitems, dictitems)
        # State is passed to __setstate__ if defined, else set in __dict__.
        return (
            self.__class__, # callable to reconstruct
            (self.x, self.y, self.z), # args for callable
            self.extra_state, # state
            None, # list items iterator (for list subclasses)
            None, # dict items iterator (for dict subclasses)
        )

    def __setstate__(self, state):
        self.extra_state = state
        # If __dict__ was part of state: self.__dict__.update(state)

    def __str__(self):
        return (
            f"ReduciblePointEx({self.x}, {self.y}, {self.z}, "
            f"extra={self.extra_state})"
        )

# Class to test memoization handling with custom __new__
# Pickle should correctly use its memo for `mt_shared_A` if it's referenced multiple times.
class MemoTricky:
    _instance_cache = {} # Class-level cache

    def __new__(cls, name):
        if name not in cls._instance_cache:
            instance = super().__new__(cls)
            instance.name = name # Initialize name here
            instance.data = [] # Mutable part specific to instance
            cls._instance_cache[name] = instance
        return cls._instance_cache[name]

    def __init__(self, name):
        # __init__ might be called on an already initialized object from cache.
        # Ensure it's idempotent or handles this.
        # For pickling, usually __new__ is called, then __setstate__ (if exists),
        # or attributes are set from __reduce__ state.
        if not hasattr(self, 'initialized_by_init'): # Avoid re-init issues
            self.initialized_by_init = True
            # self.name = name # name is set in __new__
            # self.data = [] # data is set in __new__


    def add_data(self, item):
        self.data.append(item)

    def __str__(self):
        return f"MemoTricky(name='{self.name}', data={self.data})"

    def __getstate__(self):
        # Return only what's needed to reconstruct this instance's unique state
        return {"name": self.name, "data": self.data}

    def __setstate__(self, state):
        self.name = state["name"]
        self.data = state["data"]
        # This object, after unpickling, will be a new instance,
        # not necessarily from the _instance_cache unless __reduce__ forces it.
        self.initialized_by_init = True


# Clear cache before creating test instances
MemoTricky._instance_cache.clear()
mt_shared_A = MemoTricky("shared_A")
mt_shared_A.add_data(1)
mt_shared_A.add_data(2)

mt_unique_B = MemoTricky("unique_B")
mt_unique_B.add_data(100)

# Create a list where the same MemoTricky instance appears multiple times
# This tests if pickle's internal memoization correctly handles shared references
# to custom objects during the pickling process.
memo_test_list = [mt_shared_A, mt_unique_B, mt_shared_A, MemoTricky("another_A")]
# Note: MemoTricky("another_A") will be a *new* object if "another_A" != "shared_A"
# If "another_A" was "shared_A", it would be the same as mt_shared_A.

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
MemoTricky._instance_cache.clear() # Clean up


# 7. Fuzzing Test Cases
# Configuration for the fuzzer
FUZZ_NUM_TEST_CASES = 25 # Number of fuzzed objects to generate
FUZZ_MAX_DEPTH = 3 # Max nesting depth for generated objects
FUZZ_MAX_COLLECTION_SIZE = 4 # Max elements in lists, dicts, sets
FUZZ_MAX_STRING_LENGTH = 50
FUZZ_MAX_BYTES_LENGTH = 50
FUZZ_MAX_INT = 2**30 - 1 # Keep ints a bit smaller for fuzzer
FUZZ_MIN_INT = -(2**30)

def generate_fuzzed_basic_type():
    """Generates a random basic Python type."""
    type_choice = random.choice(
        ["int", "float", "str", "bool", "none", "bytes"]
    )
    if type_choice == "int":
        return random.randint(FUZZ_MIN_INT, FUZZ_MAX_INT)
    elif type_choice == "float":
        special_floats = [float("inf"), float("-inf"), float("nan"), 0.0, -0.0]
        if random.random() < 0.2: # 20% chance of a special float
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
    else:  # 'none'
        return None

def generate_fuzzed_object(current_depth=0):
    """Recursively generates a fuzzed Python object."""
    if current_depth >= FUZZ_MAX_DEPTH:
        return generate_fuzzed_basic_type()

    # Prefer basic types more often as depth increases or by chance
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
            # Ensure keys are hashable and often basic types for simplicity
            key_type_choice = random.choice(["str_key", "int_key", "bytes_key", "bool_key", "none_key"])
            key = None
            if key_type_choice == "str_key":
                key_len = random.randint(1,10)
                key = "".join(random.choice(string.ascii_letters) for _ in range(key_len))
            elif key_type_choice == "int_key":
                key = random.randint(FUZZ_MIN_INT // 1000, FUZZ_MAX_INT // 1000)
            elif key_type_choice == "bytes_key":
                key_len = random.randint(1,10)
                key = bytes(random.getrandbits(8) for _ in range(key_len))
            elif key_type_choice == "bool_key":
                key = random.choice([True, False])
            elif key_type_choice == "none_key": # None can be a dict key
                key = None

            # Fallback for complex keys from fuzzed_object (less likely now)
            # try:
            #     hash(key)
            # except TypeError: # Should be rare with controlled key generation
            #     key = f"fallback_fuzz_key_{random.randint(0, 1000)}"
            d[key] = generate_fuzzed_object(current_depth + 1)
        return d
    elif type_choice == "set":
        size = random.randint(0, FUZZ_MAX_COLLECTION_SIZE)
        s = set()
        for _ in range(size):
            # Ensure elements are hashable for sets
            elem = generate_fuzzed_object(current_depth + 1)
            try:
                hash(elem) # Check hashability
                s.add(elem)
            except TypeError:
                # If element is not hashable, add a guaranteed hashable basic type
                s.add(generate_fuzzed_basic_type())
        return s
    return None # Should not be reached

test_cases_fuzzing = []
for i in range(FUZZ_NUM_TEST_CASES):
    fuzzed_data = None
    try:
        fuzzed_data = generate_fuzzed_object()
        desc_str = f"Fuzzed data (depth<={FUZZ_MAX_DEPTH}, size<={FUZZ_MAX_COLLECTION_SIZE})"
        # Try to get a short string representation, fallback if too complex/long
        try:
            data_repr_short = repr(fuzzed_data)[:70]
            if len(repr(fuzzed_data)) > 70:
                data_repr_short += "..."
            desc_str = f"{desc_str}: {data_repr_short}"
        except Exception:
            pass # Stick to generic description if repr fails or is too slow

        test_cases_fuzzing.append(
            {"id": f"fuzz_{i+1}", "data": fuzzed_data, "desc": desc_str}
        )
    except Exception as e:
        print(f"Error generating fuzzed test case {i+1}: {e}. Skipping this one.")
        # Optionally, add a placeholder or specific error case if desired
        test_cases_fuzzing.append(
            {
                "id": f"fuzz_{i+1}_generation_error",
                "data": None, # Or some error marker object
                "desc": f"Error during generation of fuzzed object: {e}",
            }
        )


# --- Test Execution ---
def run_tests(test_cases, category_name, protocols_to_test):
    """Runs pickle-hash tests for a list of test cases."""
    results = {}
    print(f"\n\n{'=' * 10} CATEGORY: {category_name} {'=' * 10}")

    for case in test_cases:
        case_id = case["id"]
        data = case["data"]
        desc = case["desc"]

        if "generation_error" in case_id and data is None:
            print(f"\n--- SKIPPING Case ID: {case_id} (Data generation failed) ---")
            print(f"Description: {desc}")
            results[case_id] = {"description": desc, "hashes": {"ERROR": "Data generation failed"}}
            continue

        results[case_id] = {"description": desc, "hashes": {}}
        print(f"\n--- Testing Case ID: {case_id} ---")
        print(f"Description: {desc}")
        # Data representation can be very long for fuzzed data, so limit it.
        # data_repr = repr(data)
        # print(f"Input Data Preview: {data_repr[:150]}{'...' if len(data_repr) > 150 else ''}")


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
            results[case_id]["hashes"][proto_key_name] = current_hash
            print(f"  {proto_key_name}: {current_hash}")

            # For types known to have potential instability (sets, some custom objects without
            # explicit ordering, or potentially complex fuzzed data), run a few times
            # with the default protocol to spot obvious non-determinism in *this* environment.
            if (("EXPECT POTENTIAL VARIATION" in desc.upper() or "FUZZ_" in case_id.upper())
                 and protocol is None and not current_hash.startswith("ERROR:")) :
                print(
                    f"    (Verifying potential instability for '{case_id}' with default "
                    f"protocol, 2 more runs):"
                )
                for i in range(2):  # Two more runs for this specific case
                    h = get_pickle_hash(
                        data,
                        protocol=protocol,
                        note=f"{case_id} default proto, extra run {i + 2}",
                    )
                    print(f"      Run {i + 2}: {h}")
                    if h != current_hash and not h.startswith("ERROR:"):
                        print(f"      WARNING: Hash mismatch on extra run for {case_id}!")
    return results


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

    # Determine protocols to test
    protocols_to_test_core = [None, pickle.HIGHEST_PROTOCOL]
    # Add specific common protocols if not already covered by default/highest
    # and ensure they are not higher than HIGHEST_PROTOCOL
    for p_val in [0, 1, 2, 3, 4]: # Include older protocols too for broader testing
        if p_val <= pickle.HIGHEST_PROTOCOL and p_val not in protocols_to_test_core:
            protocols_to_test_core.append(p_val)
    if sys.version_info >= (3, 8) and 5 <= pickle.HIGHEST_PROTOCOL: # Protocol 5 requires Python 3.8+
        if 5 not in protocols_to_test_core:
            protocols_to_test_core.append(5)

    # Ensure None (for default) is distinctly handled and protocols are unique and sorted.
    unique_protocols_set = set()
    processed_protocols = []

    # Add None first if it's intended to be tested (represents default)
    if None not in unique_protocols_set: # Conceptually, 'None' means default
        processed_protocols.append(None)
        unique_protocols_set.add("DefaultMarker") # Use a marker for the set

    # Add other protocols, ensuring they are valid and sorted
    # Filter out None if it was added, sort integers, then re-add None at the start.
    numeric_protocols = sorted(
        list(set(p for p in protocols_to_test_core if isinstance(p, int)))
    )

    final_protocols_to_test = []
    if None in processed_protocols: # If default was intended
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

    environment_results = {}

    for category_name, test_data_list in all_test_data_categories.items():
        environment_results[category_name] = run_tests(
            test_data_list, category_name, final_protocols_to_test
        )

    print("\n\n--- All Test Hashes Collected (for this environment) ---")
    import json # Keep import here as it's only for output

    py_version_slug = f"py{sys.version_info.major}{sys.version_info.minor}"
    platform_slug = sys.platform.replace("-", "_").lower()
    output_filename = f"pickle_hashes_{py_version_slug}_{platform_slug}.json"

    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(environment_results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {output_filename}")
    except Exception as e:
        print(f"\nError saving results to {output_filename}: {e}")

    print("\n--- Next Steps (Reminder) ---")
    print("1. Run this script in VARIOUS environments (different OS, Python versions).")
    print("2. Collect and compare the JSON output files from each environment.")
    print("3. Any differing hashes for the *same test case ID and same protocol* "
          "across environments indicate an instability.")
    print("4. This script provides a foundation. Consider deeper white-box analysis "
          "of pickle's C code if specific instabilities are found and need diagnosis.")
    print("5. Review test coverage (e.g., using a traceability matrix against pickle features).")