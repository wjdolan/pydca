"""Run all notebooks using the current Python environment."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def iter_notebooks(root: Path) -> list[Path]:
    notebooks: list[Path] = []
    for path in root.rglob("*.ipynb"):
        if ".ipynb_checkpoints" in path.parts:
            continue
        notebooks.append(path)
    return sorted(notebooks)


def build_command(notebook: Path, args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--execute",
        "--to",
        "notebook",
    ]

    if args.inplace:
        cmd.append("--inplace")
    else:
        cmd.extend(["--output-dir", str(args.output_dir)])

    if args.kernel:
        cmd.append(f"--ExecutePreprocessor.kernel_name={args.kernel}")
    if args.timeout is not None:
        cmd.append(f"--ExecutePreprocessor.timeout={args.timeout}")
    if args.allow_errors:
        cmd.append("--allow-errors")

    cmd.append(str(notebook))
    return cmd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Execute all notebooks under a root directory",
    )
    parser.add_argument(
        "--root",
        default="examples",
        help="Root directory to scan for notebooks",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite notebooks with executed output",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for executed notebooks (ignored with --inplace)",
    )
    parser.add_argument(
        "--kernel",
        default=None,
        help="Jupyter kernel name to use",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Execution timeout per notebook in seconds",
    )
    parser.add_argument(
        "--allow-errors",
        action="store_true",
        help="Continue executing cells after errors",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue to next notebook even if one fails",
    )
    args = parser.parse_args(argv)

    root = Path(args.root)
    if not root.exists():
        print(f"Root directory not found: {root}")
        return 2

    if args.inplace and args.output_dir:
        print("Use either --inplace or --output-dir, not both")
        return 2

    if not args.inplace:
        output_dir = Path(args.output_dir) if args.output_dir else root / "notebook_runs"
        output_dir.mkdir(parents=True, exist_ok=True)
        args.output_dir = output_dir

    notebooks = iter_notebooks(root)
    if not notebooks:
        print(f"No notebooks found under: {root}")
        return 0

    failures = 0
    for notebook in notebooks:
        print(f"Executing: {notebook}")
        cmd = build_command(notebook, args)
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            failures += 1
            print(f"Notebook failed: {notebook}")
            if not args.keep_going:
                return result.returncode

    print(f"Done. Executed {len(notebooks)} notebook(s) with {failures} failure(s).")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
