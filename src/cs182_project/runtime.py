from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def legacy_src_dir() -> Path:
    return project_root() / "src"


def ensure_legacy_src_on_path() -> None:
    legacy_src = str(legacy_src_dir())
    if legacy_src not in sys.path:
        sys.path.insert(0, legacy_src)


def timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(project_root()))
    except ValueError:
        return str(path)


def parse_int_list(raw: str) -> list[int]:
    values = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if chunk:
            values.append(int(chunk))
    return values


def write_overlay_config(base_config: Path, overrides: dict) -> Path:
    payload = {"inherit": [str(base_config.resolve())]}
    payload.update(overrides)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        prefix="icl-overlay-",
        delete=False,
    )
    with handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return Path(handle.name)


def discover_run_dir(base_dir: Path, before: set[str]) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    after = {path.name for path in base_dir.iterdir() if path.is_dir()}
    created = sorted(after - before)
    if created:
        return base_dir / created[-1]
    candidates = sorted(
        (path for path in base_dir.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise RuntimeError(f"No run directory was created under {base_dir}")
    return candidates[-1]


def run_command(command: list[str], env: dict | None = None) -> None:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    subprocess.run(command, cwd=project_root(), env=merged_env, check=True)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
