# compare_pickle_results.py
import json
import argparse


def compare_results(data1, data2, filename1="pickle_hashes_py36_win32.json", filename2="pickle_hashes_py311_win32.json"):
    """
    Compares two sets of pickle test results.
    Args:
        data1: Dictionary loaded from the first JSON result file.
        data2: Dictionary loaded from the second JSON result file.
        filename1: Name of the first file (for reporting).
        filename2: Name of the second file (for reporting).
    """
    mismatches_found = 0
    print(f"Comparing results from:\n  1: {filename1}\n  2: {filename2}\n")

    all_categories = set(data1.keys()) | set(data2.keys())

    for category in sorted(list(all_categories)):
        print(f"--- Category: {category} ---")

        if category not in data1:
            print(
                f"  Category exists in '{filename2}' but not in '{filename1}'. Skipping comparison for this category.")
            continue
        if category not in data2:
            print(
                f"  Category exists in '{filename1}' but not in '{filename2}'. Skipping comparison for this category.")
            continue

        cat_data1 = data1[category]
        cat_data2 = data2[category]
        all_test_ids = set(cat_data1.keys()) | set(cat_data2.keys())

        for test_id in sorted(list(all_test_ids)):
            # print(f"  Test Case ID: {test_id}") # Can be verbose

            if test_id not in cat_data1:
                print(f"    Test ID '{test_id}' exists in '{filename2}' but not in '{filename1}'.")
                mismatches_found += 1
                continue
            if test_id not in cat_data2:
                print(f"    Test ID '{test_id}' exists in '{filename1}' but not in '{filename2}'.")
                mismatches_found += 1
                continue

            case1 = cat_data1[test_id]
            case2 = cat_data2[test_id]

            # Optional: Compare descriptions if needed, though hashes are key
            # if case1.get("description") != case2.get("description"):
            # print(f"    Description mismatch for '{test_id}' (not critical for hash comparison)")

            hashes1 = case1.get("hashes", {})
            hashes2 = case2.get("hashes", {})
            all_protocols = set(hashes1.keys()) | set(hashes2.keys())

            has_id_mismatch = False
            for protocol in sorted(list(all_protocols)):
                hash1 = hashes1.get(protocol)
                hash2 = hashes2.get(protocol)

                if protocol not in hashes1:
                    if not has_id_mismatch: print(f"  Test Case ID: {test_id} ({case1.get('description', '')[:50]}...)")
                    print(
                        f"    Protocol '{protocol}': Exists in '{filename2}' (Hash: {hash2}) but MISSING in '{filename1}'")
                    has_id_mismatch = True
                    mismatches_found += 1
                    continue
                if protocol not in hashes2:
                    if not has_id_mismatch: print(f"  Test Case ID: {test_id} ({case1.get('description', '')[:50]}...)")
                    print(
                        f"    Protocol '{protocol}': Exists in '{filename1}' (Hash: {hash1}) but MISSING in '{filename2}'")
                    has_id_mismatch = True
                    mismatches_found += 1
                    continue

                if hash1 != hash2:
                    if not has_id_mismatch: print(f"  Test Case ID: {test_id} ({case1.get('description', '')[:50]}...)")
                    print(f"    Protocol '{protocol}': MISMATCH!")
                    print(f"      {filename1}: {hash1}")
                    print(f"      {filename2}: {hash2}")
                    has_id_mismatch = True
                    mismatches_found += 1
                # else:
                # print(f"    Protocol '{protocol}': MATCH ({hash1})") # Usually too verbose

            if has_id_mismatch:
                print()  # Add a newline for readability after a test case with mismatches

    if mismatches_found == 0:
        print("\n+++ All compared hashes and structures are IDENTICAL. +++")
    else:
        print(f"\n--- Found {mismatches_found} mismatches/differences. ---")


def main():
    parser = argparse.ArgumentParser(description="Compare two pickle test result JSON files.")
    parser.add_argument("file1", help="Path to the first JSON result file.")
    parser.add_argument("file2", help="Path to the second JSON result file.")
    args = parser.parse_args()

    try:
        with open(args.file1, "r", encoding="utf-8") as f1:
            data1 = json.load(f1)
        print(f"Successfully loaded: {args.file1}")
    except FileNotFoundError:
        print(f"Error: File not found: {args.file1}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {args.file1}")
        return

    try:
        with open(args.file2, "r", encoding="utf-8") as f2:
            data2 = json.load(f2)
        print(f"Successfully loaded: {args.file2}")
    except FileNotFoundError:
        print(f"Error: File not found: {args.file2}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {args.file2}")
        return

    compare_results(data1, data2, args.file1, args.file2)


if __name__ == "__main__":
    main()