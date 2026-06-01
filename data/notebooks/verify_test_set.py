"""
verify_test_set.py
==================
Run this BEFORE every evaluation run to confirm the test set has not
been modified since it was locked.

Usage
-----
    python verify_test_set.py                         # uses data/ defaults
    python verify_test_set.py --data_dir data/
    python verify_test_set.py --test_set path/to/test_set.jsonl \
                               --lock    path/to/test_set.lock

Exit codes
----------
    0  — test set is intact, safe to evaluate
    1  — INTEGRITY FAILURE — test set has been modified, do not evaluate
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(test_set_path: str, lock_path: str) -> bool:
    # Load lock file
    if not Path(lock_path).exists():
        print(f"[ERROR] Lock file not found: {lock_path}")
        print("        Run build_test_set.py first to create the lock.")
        return False

    with open(lock_path) as f:
        lock = json.load(f)

    expected_digest = lock["sha256"]
    locked_at       = lock["locked_at"]
    expected_count  = lock["num_questions"]

    # Verify file exists
    if not Path(test_set_path).exists():
        print(f"[ERROR] Test set file not found: {test_set_path}")
        return False

    # Check SHA-256
    actual_digest = sha256_of_file(test_set_path)
    if actual_digest != expected_digest:
        print("=" * 60)
        print("  ❌  INTEGRITY FAILURE")
        print(f"  Expected SHA-256 : {expected_digest}")
        print(f"  Actual SHA-256   : {actual_digest}")
        print(f"  Locked at        : {locked_at}")
        print()
        print("  The test set has been modified since locking.")
        print("  Evaluation results would be INVALID.")
        print("  To regenerate a clean test set, run build_test_set.py.")
        print("=" * 60)
        return False

    # Count questions
    with open(test_set_path) as f:
        actual_count = sum(1 for line in f if line.strip())

    if actual_count != expected_count:
        print(f"[WARN] Question count mismatch: expected {expected_count}, found {actual_count}")
        print("       (SHA-256 matched — file content is identical, this is likely a counting edge case)")

    print("=" * 60)
    print("  ✅  TEST SET INTEGRITY VERIFIED")
    print(f"  SHA-256       : {actual_digest}")
    print(f"  Questions     : {actual_count}")
    print(f"  Locked at     : {locked_at}")
    print(f"  Test set path : {test_set_path}")
    print("=" * 60)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",  default="data",                  help="Directory containing test_set.jsonl and test_set.lock")
    parser.add_argument("--test_set",  default=None,                    help="Override path to test_set.jsonl")
    parser.add_argument("--lock",      default=None,                    help="Override path to test_set.lock")
    args = parser.parse_args()

    test_set_path = args.test_set or f"{args.data_dir}/test_set.jsonl"
    lock_path     = args.lock     or f"{args.data_dir}/test_set.lock"

    ok = verify(test_set_path, lock_path)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
