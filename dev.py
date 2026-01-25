#!/usr/bin/env python3
import argparse
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


def get_runner():
    """Detects if uv is available, otherwise falls back to python -m."""
    if shutil.which("uv"):
        return "uv run"
    return f"{sys.executable} -m"


RUNNER = get_runner()


def confirm_and_run(cmd: str, use_runner: bool = True, skip_confirm: bool = False):
    """Builds, displays, and confirms a command before running."""
    full_cmd = f"{RUNNER} {cmd}" if use_runner else cmd

    print(f"\nProposed command: {full_cmd}")

    if not skip_confirm:
        response = input("Run this command? (y/n): ").lower().strip()
        if response != "y":
            print("Skipped.")
            return

    try:
        subprocess.run(shlex.split(full_cmd), check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed with exit code {e.returncode}")
        sys.exit(e.returncode)


# Subcommands
def setup(skip):
    if "uv run" in RUNNER:
        confirm_and_run("uv sync --all-groups --all-extras", use_runner=False, skip_confirm=skip)
    else:
        confirm_and_run("pip install -e .[dev,test,docs]", use_runner=False, skip_confirm=skip)
    confirm_and_run("pre-commit install", skip_confirm=skip)


def docs_clean(skip):
    path = Path("docs/_build")
    print(f"\nProposed action: Delete {path}")
    if skip or input("Proceed with cleanup? (y/n): ").lower().strip() == "y":
        if path.exists():
            shutil.rmtree(path)
            print("Done: docs/_build removed.")
        else:
            print("Nothing to clean.")


def main():
    parser = argparse.ArgumentParser(
        description="NeuroKit Development CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: ./dev.py test",
    )

    # Define tasks
    subparsers = parser.add_subparsers(dest="task", help="The task to execute")

    subparsers.add_parser("setup", help="Install dependencies and hooks")
    subparsers.add_parser("test", help="Run tests via pytest")
    subparsers.add_parser("lint", help="Check linting with ruff")
    subparsers.add_parser("format", help="Format code with ruff")
    subparsers.add_parser("docs", help="Build HTML documentation")
    subparsers.add_parser("docs-clean", help="Remove documentation build artifacts")
    subparsers.add_parser("docs-serve", help="Serve documentation on localhost:8000")

    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompts")

    # Help check: If no arguments are passed, print help and exit
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    # Mapping tasks to commands
    if args.task == "setup":
        setup(args.yes)
    elif args.task == "test":
        confirm_and_run("pytest", skip_confirm=args.yes)
    elif args.task == "lint":
        confirm_and_run("ruff check", skip_confirm=args.yes)
    elif args.task == "format":
        confirm_and_run("ruff format --check", skip_confirm=args.yes)
    elif args.task == "docs":
        confirm_and_run("sphinx-build -j auto -b html docs/ docs/_build/html", skip_confirm=args.yes)
    elif args.task == "docs-clean":
        docs_clean(args.yes)
    elif args.task == "docs-serve":
        confirm_and_run("python -m http.server --directory docs/_build/html 8000", skip_confirm=args.yes)


if __name__ == "__main__":
    main()
