import os

from fastapi.testclient import TestClient


def test_api_flow(tmp_path):
    os.environ["DORSAL_EVIDENCE_DIR"] = str(tmp_path)
    import importlib
    import dorsal.api as api
    importlib.reload(api)
    c = TestClient(api.app)
    s = c.post("/api/review").json()
    assert s["briefing"] and s["plan"]["steps"]
    fin = next(x for x in s["plan"]["steps"] if x["tier"] == "borrow")
    assert c.post(f"/api/actions/{fin['id']}/approve", json={"pin": "0000"}).status_code == 400
    assert c.post(f"/api/actions/{fin['id']}/approve", json={"pin": "2468"}).status_code == 200
    atk = c.post("/api/attack").json()
    assert all(a["status"] == "denied" for a in atk["attempts"])
    assert c.get("/api/audit/verify").json()["ok"]
    assert c.post("/api/settings", json={"auto_cost_limit": 99999}).status_code == 400
