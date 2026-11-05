"""README.md stays in step with the program: the command-line reference is complete and current."""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_cli_reference_is_current():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    m = re.search(r"<!-- cli-reference:start -->\n(.*?)\n<!-- cli-reference:end -->", readme, re.S)
    assert m, "README has no command-line reference section"
    ref = subprocess.run([sys.executable, "tools/cli_reference.py"], cwd=ROOT, capture_output=True, text=True,
                         check=True).stdout.strip()
    assert m.group(1).strip() == ref, "run: python tools/cli_reference.py and paste it into README.md"


def test_every_option_has_help():
    ref = subprocess.run([sys.executable, "tools/cli_reference.py"], cwd=ROOT, capture_output=True, text=True,
                         check=True).stdout
    empty = [line for line in ref.splitlines() if line.startswith("| `") and line.rstrip().endswith("|  |")]
    assert not empty, empty


def test_cli_reference_shows_help_as_the_command_line_does():
    """argparse's %(default)s and %% are expanded, as `totality ... --help` prints them (they were
    shown raw in the README, e.g. `--achf-weights ... (default %(default)s)`)."""
    import sys

    sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
    from cli_reference import reference

    text = reference()
    assert "%(" not in text and "%%" not in text
    assert "in % of the Sun's radius" in text
