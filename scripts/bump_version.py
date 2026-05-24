#!/usr/bin/env python3
"""Bump the patch version in pyproject.toml before each commit."""

import re
import sys

from pathlib import Path


PYPROJECT = Path(__file__).parent.parent / "pyproject.toml"


content = PYPROJECT.read_text(encoding="utf-8")
match = re.search(r'^(version\s*=\s*")(\d+)\.(\d+)\.(\d+)(")', content, re.MULTILINE)

if not match:
    print("bump_version: could not find version field in pyproject.toml", file=sys.stderr)
    sys.exit(1)

major, minor, patch = int(match.group(2)), int(match.group(3)), int(match.group(4))
new_version = f"{major}.{minor}.{patch + 1}"
new_content = content[: match.start()] + f'{match.group(1)}{new_version}{match.group(5)}' + content[match.end():]

PYPROJECT.write_text(new_content, encoding="utf-8")
print(f"bump_version: {major}.{minor}.{patch} -> {new_version}")
