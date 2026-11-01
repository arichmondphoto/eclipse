"""Write the command-line option reference (Markdown) from the parser itself.

    python tools/cli_reference.py > /tmp/ref.md

README.md's "Command-line reference" section is this output; tests/test_readme.py checks it
is up to date.
"""

import argparse
import sys
from unittest import mock

sys.path.insert(0, "src")


class _Got(Exception):
    pass


def parser() -> argparse.ArgumentParser:
    from tse_core import cli

    box = {}

    def grab(self, *a, **k):
        box["p"] = self
        raise _Got

    with mock.patch.object(argparse.ArgumentParser, "parse_args", grab):
        try:
            cli.main([])
        except _Got:
            pass
    return box["p"]


def reference() -> str:
    p = parser()
    sub = next(a for a in p._actions if isinstance(a, argparse._SubParsersAction))
    helps = {c.dest: c.help or "" for c in sub._choices_actions}
    out = []
    for name, sp in sub.choices.items():
        out.append(f"### `totality {name}`\n")
        if helps.get(name):
            out.append(helps[name].rstrip(".") + ".\n")
        rows = []
        for a in sp._actions:
            if isinstance(a, argparse._HelpAction):
                continue
            if a.option_strings:
                flag = ", ".join(f"`{o}`" for o in a.option_strings)
            else:
                flag = f"`{a.dest}`"
            h = (a.help or "").replace("|", "\\|").replace("\n", " ")
            if a.default not in (None, False, argparse.SUPPRESS, "") and "default" not in h and a.option_strings:
                h = (h + f" (default {a.default})").strip()
            rows.append(f"| {flag} | {h} |")
        if rows:
            out.append("| Option | What it does |\n| --- | --- |\n" + "\n".join(rows) + "\n")
    return "\n".join(out)


if __name__ == "__main__":
    print(reference())
