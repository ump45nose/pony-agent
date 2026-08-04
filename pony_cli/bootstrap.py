"""Public Pony environment mapped into the retained legacy implementation."""

from __future__ import annotations

import os
from pathlib import Path


_PONY_ONLY = {"PONY_AGENT_CORE", "PONY_BOOTSTRAPPED"}


def _load_pony_env_file(pony_home: Path) -> None:
    env_file = pony_home / ".env"
    if not env_file.is_file():
        return
    try:
        from dotenv import dotenv_values

        values = dotenv_values(env_file)
    except Exception:
        return
    for key, value in values.items():
        if key and key.startswith("PONY_") and value is not None:
            os.environ.setdefault(key, value)


def bootstrap_legacy_environment() -> Path:
    """Ignore old public Hermes state, then prepare private compatibility vars."""
    if os.environ.get("PONY_BOOTSTRAPPED") == "1":
        return Path(os.environ["PONY_HOME"])

    for key in tuple(os.environ):
        if key.startswith("HERMES_"):
            os.environ.pop(key, None)

    pony_home = Path(os.environ.get("PONY_HOME") or (Path.home() / ".pony")).expanduser()
    os.environ["PONY_HOME"] = str(pony_home)
    _load_pony_env_file(pony_home)

    for key, value in tuple(os.environ.items()):
        if key.startswith("PONY_") and key not in _PONY_ONLY:
            os.environ["HERMES_" + key.removeprefix("PONY_")] = value
    os.environ["HERMES_HOME"] = str(pony_home)
    os.environ["PONY_BOOTSTRAPPED"] = "1"
    return pony_home
