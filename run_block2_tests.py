#!/usr/bin/env python
"""
Block 2 Verification Runner
Runs all safety layer tests and reports status.

Usage: python run_block2_tests.py
"""

import subprocess
import sys
import os


def run_test(script_path: str, description: str) -> bool:
    """Run a test script and report results."""
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"Script: {script_path}")
    print('='*70)

    try:
        # Ensure we run from project root
        project_root = os.path.dirname(os.path.abspath(__file__))
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=project_root,
            capture_output=False,
            text=True,
        )
        return result.returncode == 0
    except Exception as e:
        print(f"Error running {script_path}: {e}")
        return False


def main():
    """Run Block 2 verification suite."""
    
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  BLOCK 2 COMPREHENSIVE TEST SUITE".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)

    tests = [
        ("tests/test_safety.py", "Medical Safety Layer Tests"),
    ]

    results = []

    for script, description in tests:
        passed = run_test(script, description)
        results.append((description, passed))

    # Summary
    print("\n" + "="*70)
    print("BLOCK 2 TEST SUITE SUMMARY")
    print("="*70)

    for description, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {description:.<50} {status}")

    all_passed = all(result for _, result in results)

    print("\n" + "="*70)
    if all_passed:
        print("✓✓✓ BLOCK 2 FULLY VERIFIED ✓✓✓")
        print("="*70)
        print("\nThe biomedical safety layer is operational and tested.")
        print("Ready to proceed to Block 3: Retrieval Grounding")
        return 0
    else:
        print("✗✗✗ BLOCK 2 VERIFICATION INCOMPLETE ✗✗✗")
        print("="*70)
        print("\nReview failed tests above before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
