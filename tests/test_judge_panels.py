"""Judge panel selection: PWB_JUDGE_PANEL picks an alternative panel from config.yaml."""
from pathlib import Path

import pytest
import yaml

from src.judge import build_judge_panel

CONFIG = yaml.safe_load((Path(__file__).resolve().parent.parent / "config.yaml").read_text())


def _ids(panel):
    return [j.client.config.id for j in panel.judges]


def test_default_panel_is_the_published_one(monkeypatch):
    monkeypatch.delenv("PWB_JUDGE_PANEL", raising=False)
    assert _ids(build_judge_panel(CONFIG)) == [j["id"] for j in CONFIG["judges"]]


def test_openrouter_panel_keeps_the_same_three_families(monkeypatch):
    monkeypatch.setenv("PWB_JUDGE_PANEL", "openrouter")
    ids = _ids(build_judge_panel(CONFIG))
    assert len(ids) == 3
    assert all(i.startswith("openrouter/") for i in ids)
    assert any("devstral" in i for i in ids)
    assert any("gpt-oss-120b" in i for i in ids)
    assert any("gemma-4-31b" in i for i in ids)


def test_unknown_panel_fails_loudly(monkeypatch):
    monkeypatch.setenv("PWB_JUDGE_PANEL", "nope")
    with pytest.raises(ValueError, match="not in judge_panels"):
        build_judge_panel(CONFIG)


def test_openrouter_models_use_the_openrouter_key(monkeypatch):
    monkeypatch.setenv("PWB_JUDGE_PANEL", "openrouter")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    panel = build_judge_panel(CONFIG)
    assert {j.client._resolve_api_key() for j in panel.judges} == {"openrouter-key"}
