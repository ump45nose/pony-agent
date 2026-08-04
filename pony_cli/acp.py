"""`pony-acp` compatibility entrypoint."""

from __future__ import annotations

from pony_cli.bootstrap import bootstrap_legacy_environment


def main() -> None:
    bootstrap_legacy_environment()
    from acp_adapter.entry import main as legacy_main

    legacy_main()
