#!/usr/bin/env python
# ruff: noqa: T201

from ci_helpers import CI
from ci_helpers.git import Diff
from pathlib import Path
import asyncio
import sys


async def main():
    ci = CI()
    diff = Diff(Path(sys.argv[1]).read_text())
    async for format_diff in ci.ruff.format_diffs(diff):
        print(format_diff)


if __name__ == "__main__":
    asyncio.run(main())
