#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["packaging"]
# ///

"""
upgrade-deps.py - Raise the dependency floors of PEP 723 scripts.

The typeout scripts ship without a lockfile (they are embedded into the
amalgamation), so the inline metadata is the only thing that decides what
users get. This does the moral equivalent of `uv lock --upgrade`:

1. Resolve each script from scratch with `uv lock --script --upgrade`
   (on a temporary copy, so no lockfile is left behind).
2. For every dependency that already carries a `>=` floor, raise that
   floor to the newly resolved version. Upper bounds and exclusions are
   left alone, and unconstrained dependencies stay unconstrained.
3. Report dependencies whose latest release is blocked by an upper bound,
   since widening those is a manual decision.

Changing the metadata also changes uv's cache key for the script
environment, so existing installs re-resolve on their next run.

Usage: ./upgrade-deps.py [--dry-run] SCRIPT...
"""

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

DEP_LINE = re.compile(r'^(#\s+)"(?P<req>[^"]+)"(,?\s*)$')


def resolve(script: Path) -> tuple[dict[str, Version], str]:
    """Lock a temporary copy of script; return resolved versions and outdated report."""
    with tempfile.TemporaryDirectory(prefix="typeout-upgrade-") as tmpdir:
        tmp = Path(tmpdir) / script.name
        shutil.copy2(script, tmp)
        subprocess.run(["uv", "lock", "--script", str(tmp), "--upgrade", "--quiet"], check=True)
        lock = tomllib.loads(tmp.with_name(tmp.name + ".lock").read_text())
        tree = subprocess.run(
            ["uv", "tree", "--script", str(tmp), "--outdated", "--depth", "0", "--quiet"],
            check=True, capture_output=True, text=True,
        ).stdout

    # A package can appear several times with per-platform versions;
    # the floor must hold everywhere, so keep the lowest.
    versions: dict[str, Version] = {}
    for pkg in lock.get("package", []):
        name = canonicalize_name(pkg["name"])
        v = Version(pkg["version"])
        versions[name] = min(versions.get(name, v), v)
    return versions, tree


def _held_back(tree_line: str) -> bool:
    """True for a `uv tree --outdated` line whose latest release is really newer.

    Ignores local-version-only differences such as torch 2.14.0 vs 2.14.0+cpu.
    """
    m = re.search(r" v(\S+) \(latest: v([^)]+)\)", tree_line)
    return bool(m) and Version(m[2]).public != Version(m[1]).public


def upgrade(script: Path, dry_run: bool) -> None:
    print(f"==> {script}")
    versions, tree = resolve(script)

    lines = script.read_text().splitlines(keepends=True)
    in_deps = False
    changed = False
    for i, line in enumerate(lines):
        if line.startswith("# dependencies = ["):
            in_deps = True
            continue
        if in_deps and line.startswith("# ]"):
            break
        m = DEP_LINE.match(line) if in_deps else None
        if not m:
            continue

        req = Requirement(m["req"])
        resolved = versions.get(canonicalize_name(req.name))
        if resolved is None:
            continue
        for spec in req.specifier:
            if spec.operator != ">=" or Version(spec.version) >= resolved:
                continue
            new_req = m["req"].replace(f">={spec.version}", f">={resolved}", 1)
            print(f"    {m['req']}  ->  {new_req}")
            lines[i] = line.replace(m["req"], new_req, 1)
            changed = True

    if not changed:
        print("    floors already current")
    elif not dry_run:
        script.write_text("".join(lines))

    held = [l.strip() for l in tree.splitlines() if _held_back(l)]
    if held:
        print("    held back by upper bounds or other packages (review manually):")
        for l in held:
            print(f"      {l}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("scripts", nargs="+", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="report, do not rewrite")
    args = parser.parse_args()

    if not shutil.which("uv"):
        sys.exit("error: uv is required")
    for script in args.scripts:
        upgrade(script, args.dry_run)


if __name__ == "__main__":
    main()
