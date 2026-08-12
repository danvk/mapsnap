"""Tests for the command registry itself."""

import importlib
import sys

import pytest

from mapsnap.cli import SUBCOMMANDS


@pytest.mark.parametrize("name", sorted(SUBCOMMANDS))
def test_every_command_module_imports(name: str) -> None:
    """Each registered command's module imports on its own and has a main().

    Every mapsnap.* module is evicted from sys.modules first, so each command
    is imported the way its own invocation would: first, into a clean
    interpreter. That is what makes this catch a circular import.

    `mapsnap fit` broke exactly this way -- fit imported archive while archive
    imported fit -- and nothing caught it. Unit tests import leaf functions, not
    whole modules; and an earlier version of this test imported modules in
    alphabetical order, where `archive` lands first and resolves the cycle
    before `fit` is reached. The break only appears when the module that owns
    the cycle is imported first, which is precisely what running the command
    does.
    """
    for module_name in [key for key in sys.modules if key.startswith("mapsnap")]:
        del sys.modules[module_name]

    module_name, _ = SUBCOMMANDS[name]
    module = importlib.import_module(module_name)
    assert callable(getattr(module, "main", None)), (
        f"{module_name} has no main() for `mapsnap {name}`"
    )


def test_fatal_signal_produces_a_python_traceback():
    """A native crash must name the Python line, not just exit 245.

    #296 has killed `snap` three times in ~60 fits and every report was a bare
    exit code, because a SIGSEGV inside numpy/shapely/cv2/torch unwinds no
    Python frames. faulthandler writes both stacks from the signal handler.
    """
    import subprocess
    import sys

    crash = (
        "import ctypes\n"
        "from mapsnap import cli\n"
        "cli.faulthandler.enable()\n"
        "ctypes.string_at(0)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", crash], capture_output=True, text=True, check=False
    )
    assert result.returncode != 0
    assert "Fatal Python error" in result.stderr
    assert "ctypes" in result.stderr
