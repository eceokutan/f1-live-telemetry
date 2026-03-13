"""
Tests for deterministic/safe Jarvis Post discovery behavior.
"""

from pathlib import Path
import sys

import analysis.ai_pipeline_bridge as bridge_module
from analysis.ai_pipeline_bridge import AIPipelineBridge


def _new_bridge() -> AIPipelineBridge:
    bridge = AIPipelineBridge.__new__(AIPipelineBridge)
    bridge._external_source = ""
    return bridge


def test_candidate_roots_defaults_to_repo_root(monkeypatch):
    monkeypatch.delenv("JARVIS_POST_ROOT", raising=False)
    bridge = _new_bridge()

    roots = bridge._get_jarvis_post_candidate_roots()
    repo_root = Path(bridge_module.__file__).resolve().parent.parent

    assert roots == [repo_root]


def test_candidate_roots_prefers_explicit_env_root(monkeypatch):
    explicit_root = Path.cwd().resolve()
    monkeypatch.setenv("JARVIS_POST_ROOT", str(explicit_root))

    bridge = _new_bridge()
    roots = bridge._get_jarvis_post_candidate_roots()
    repo_root = Path(bridge_module.__file__).resolve().parent.parent

    assert roots[0] == explicit_root
    assert repo_root in roots


def test_temporary_sys_path_restores_after_context():
    bridge = _new_bridge()
    temp_root = Path.cwd().resolve()
    before = list(sys.path)

    with bridge._temporary_sys_path(temp_root):
        assert str(temp_root) in sys.path

    assert sys.path == before


def test_discovery_loads_first_valid_root_only():
    bridge = _new_bridge()
    invalid_root = Path("C:/invalid/root")
    valid_root = Path.cwd().resolve()
    loaded_roots: list[Path] = []

    bridge._get_jarvis_post_candidate_roots = lambda: [invalid_root, valid_root]
    bridge._is_valid_jarvis_post_root = lambda root: root == valid_root

    def _fake_load(root: Path):
        loaded_roots.append(root)
        return {
            "kind": "jarvis_post",
            "root": root,
            "race_agent_cls": object,
            "coach_agent_cls": object,
            "llm_client_cls": object,
        }

    bridge._load_jarvis_post_handler = _fake_load

    handler = bridge._discover_jarvis_post_pipeline()

    assert handler is not None
    assert loaded_roots == [valid_root]
    assert bridge._external_source == f"jarvis_post:{valid_root}"
