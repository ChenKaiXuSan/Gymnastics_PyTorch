"""Small dispatcher shared by the four stage entry points.

Each stage package exposes ``python -m <stage> <command> [args]``; this helper
turns a ``{command: (module, function, accepts_argv)}`` table into that
interface and prepares the runtime cache directories under ``local/``.
"""

from __future__ import annotations

import argparse
from importlib import import_module
import os
from pathlib import Path
import sys
from typing import Mapping, Sequence

from common.paths import LOCAL_ROOT

CommandTable = Mapping[str, tuple[str, str, bool, str]]
"""``name -> (module, function, accepts_argv, help)``."""


def configure_runtime_cache() -> None:
    """Keep Hugging Face, matplotlib and XDG caches below ``local/cache``."""
    credential_cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    os.environ.setdefault("HF_HOME", str(credential_cache / "huggingface"))
    cache_root = LOCAL_ROOT / "cache"
    for name in ("matplotlib", "xdg"):
        (cache_root / name).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))


def invoke(target: tuple[str, str, bool, str], prog: str, argv: list[str]) -> int:
    module_name, function_name, accepts_argv, _ = target
    function = getattr(import_module(module_name), function_name)
    if accepts_argv:
        result = function(argv)
    else:
        previous = sys.argv
        sys.argv = [prog, *argv]
        try:
            result = function()
        finally:
            sys.argv = previous
    return int(result) if isinstance(result, int) else 0


def dispatch(
    prog: str,
    description: str,
    commands: CommandTable,
    argv: Sequence[str] | None = None,
    *,
    default: str | None = None,
) -> int:
    """Parse ``<command> [args]`` and run the matching entry.

    ``default`` names the command used when none is given (stages with a
    single main action). Everything after the command is passed through to
    the target untouched, so each target keeps its own ``--help``.
    """
    configure_runtime_cache()
    parser = argparse.ArgumentParser(prog=prog, description=description)
    subparsers = parser.add_subparsers(dest="command")
    for name, (_, _, _, help_text) in commands.items():
        subparsers.add_parser(name, help=help_text, add_help=False)
    args, remainder = parser.parse_known_args(list(argv) if argv is not None else None)
    command = args.command or default
    if command is None:
        parser.print_help()
        return 0
    return invoke(commands[command], f"{prog} {command}", remainder)
