from pathlib import Path

import yaml

from cs182_project.runtime import parse_int_list, write_overlay_config


def test_parse_int_list():
    assert parse_int_list("0, 2,4") == [0, 2, 4]


def test_write_overlay_config(tmp_path: Path):
    base_config = tmp_path / "base.yaml"
    base_config.write_text("seed: 1\n", encoding="utf-8")
    overlay_path = write_overlay_config(base_config, {"seed": 3, "device": "cpu"})
    payload = yaml.safe_load(overlay_path.read_text(encoding="utf-8"))
    assert payload["inherit"] == [str(base_config.resolve())]
    assert payload["seed"] == 3
    assert payload["device"] == "cpu"
