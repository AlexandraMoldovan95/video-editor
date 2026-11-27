"""The missing-key check: the AI routes (plan, variants, deep scan) report a missing Anthropic API
key at once, before any job starts, instead of extracting every clip's frames first.

No real API call and no .env: load_dotenv is replaced by a no-op, and the key is removed from (or
set in) the environment for each test only.

Run from the repo root:  python -m pytest -q
"""
import json

import pytest
from fastapi.testclient import TestClient

from app import planner


@pytest.fixture
def no_dotenv(monkeypatch):
    # api_key_configured() imports load_dotenv when it runs, so patching the module is enough
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)


def test_no_key_is_reported(no_dotenv, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert planner.api_key_configured() is False


def test_a_key_is_seen(no_dotenv, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-not-a-real-key")
    assert planner.api_key_configured() is True


def test_make_client_refuses_without_a_key(no_dotenv, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY is missing"):
        planner.make_client()


@pytest.mark.parametrize("path", ["/api/projects/p0/plan", "/api/projects/p0/plan/variants",
                                  "/api/projects/p0/deepscan/0"])
def test_ai_routes_answer_at_once_without_a_key(no_dotenv, monkeypatch, path):
    """503 with the same message the page turns into plain words, and no job is started
    (the check runs before the project folder is even looked up)."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from app import main
    started = []
    monkeypatch.setattr(main, "run_job", lambda *a, **k: started.append(a) or "job")
    r = TestClient(main.app).post(path)
    assert r.status_code == 503
    assert r.json()["detail"] == planner.MISSING_KEY
    assert started == []


@pytest.mark.parametrize("key, ready", [(None, False), ("test-not-a-real-key", True)])
def test_config_says_whether_the_ai_is_ready(no_dotenv, monkeypatch, key, ready):
    """The page hides the AI-only controls in simple mode when this is false; the key itself
    never leaves the server."""
    if key is None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    else:
        monkeypatch.setenv("ANTHROPIC_API_KEY", key)
    from app import main
    cfg = TestClient(main.app).get("/api/config").json()
    assert cfg["ai_ready"] is ready
    assert "test-not-a-real-key" not in json.dumps(cfg)
