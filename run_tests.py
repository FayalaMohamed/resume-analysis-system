#!/usr/bin/env python3
"""Script to run all tests in the tests folder."""

import subprocess
import sys


def main():
    result = subprocess.run(
        ["python", "-m", "pytest", "tests/", "-v"],
        capture_output=False,
        text=True,
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
