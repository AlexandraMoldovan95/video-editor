"""Project names and the output format (vertical 9:16, square 1:1, horizontal 16:9).

The routes run in-process (FastAPI's TestClient) against a temporary projects folder, so the real
projects/ folder is never touched. No ffmpeg, no API key.

Run from the repo root:  python -m pytest -q
"""
import json

import pytest
from fastapi.testclient import TestClient

from app import main


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "PROJECTS", tmp_path)
    return TestClient(main.app)


def new_project(client) -> str:
    return client.post("/api/projects").json()["id"]


# ---------- names ----------

@pytest.mark.parametrize("raw, clean", [
    ("  Hike with Ana  ", "Hike with Ana"),
    ("Bakery\n\tspring   menu", "Bakery spring menu"),
    ("x" * 100, "x" * 60),
    ("", ""),
    ("\x00\x07", ""),
])
def test_project_names_are_cleaned(raw, clean):
    assert main.clean_project_name(raw) == clean


def test_a_named_project_shows_its_name_in_the_list(client):
    pid = new_project(client)
    assert client.put(f"/api/projects/{pid}/name", json={"name": "  Hike with Ana "}).json() == {"name": "Hike with Ana"}
    listed = {p["id"]: p for p in client.get("/api/projects").json()}
    assert listed[pid]["name"] == "Hike with Ana"
    assert client.get(f"/api/projects/{pid}").json()["name"] == "Hike with Ana"


def test_an_unnamed_project_lists_an_empty_name(client):
    pid = new_project(client)
    assert {p["id"]: p for p in client.get("/api/projects").json()}[pid]["name"] == ""


# ---------- output format ----------

def test_with_format_sets_only_the_size():
    edl = {"output": {"width": 1080, "height": 1920, "fps": 25}, "clips": [{"source": "a.mp4"}]}
    out = main.with_format(edl, "16:9")
    assert out["output"] == {"width": 1920, "height": 1080, "fps": 25}
    assert out["clips"] == edl["clips"] and edl["output"]["width"] == 1080   # the original isn't changed


def test_a_new_project_is_vertical(client):
    pid = new_project(client)
    assert client.get(f"/api/projects/{pid}").json()["format"] == "9:16"


def test_changing_the_format_resizes_the_current_plan(client, tmp_path):
    pid = new_project(client)
    state = tmp_path / pid / "state.json"
    st = json.loads(state.read_text("utf-8"))
    st["edl"] = {"output": {"width": 1080, "height": 1920, "fps": 30}, "clips": [{"source": "a.jpg", "duration": 2.5}]}
    state.write_text(json.dumps(st), "utf-8")
    r = client.put(f"/api/projects/{pid}/format", json={"format": "1:1"}).json()
    assert r["format"] == "1:1" and r["edl"]["output"] == {"width": 1080, "height": 1080, "fps": 30}
    again = client.get(f"/api/projects/{pid}").json()
    assert again["format"] == "1:1" and again["edl"]["output"]["width"] == 1080 and again["edl"]["output"]["height"] == 1080


def test_an_unknown_format_is_refused(client):
    pid = new_project(client)
    r = client.put(f"/api/projects/{pid}/format", json={"format": "4:5"})
    assert r.status_code == 422
    assert client.get(f"/api/projects/{pid}").json()["format"] == "9:16"
