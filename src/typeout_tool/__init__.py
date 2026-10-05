"""Console entry point for the packaged typeout script."""

from __future__ import annotations

import os
import shutil
import sys
from importlib.resources import as_file, files


def main() -> None:
    """Run the bundled typeout shell script."""
    script = files(__package__).joinpath("typeout")
    with as_file(script) as script_path:
        bash = shutil.which("bash") or "/bin/bash"
        os.execv(bash, ["bash", str(script_path), *sys.argv[1:]])


if __name__ == "__main__":
    main()
