"""
Synthetic Shopper Swarm — FastAPI backend.

Serves the UI and two JSON endpoints:
  POST /api/simulate  -> run the swarm (fast, deterministic maths)
  POST /api/narrate   -> re-run the same scenario and add the LLM "human truth"

The narrate endpoint recomputes from the same (scenario, promo, population, seed)
inputs so it is fully reproducible and never trusts client-side numbers.
"""
from __future__ import annotations

import json
import pathlib

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

import swarm
from choice import PROMO_TYPES
from explain import narrate
from personas import generate_population, segment_table

BASE = pathlib.Path(__file__).parent
SHELF = json.loads((BASE / "data" / "shelf.json").read_text(encoding="utf-8"))
SCENARIOS = {s["id"]: s for s in SHELF["scenarios"]}

app = FastAPI(title="Synthetic Shopper Swarm")

_pop_cache: dict[tuple[int, int], list] = {}


def population(n: int, seed: int):
    key = (n, seed)
    if key not in _pop_cache:
        _pop_cache[key] = generate_population(n, seed)
    return _pop_cache[key]


class SimReq(BaseModel):
    scenario_id: str
    target: str
    promo: dict               # {"type": ..., "depth": ...}
    population: int = 600
    seed: int = 7


def _run(req: SimReq) -> dict:
    sc = SCENARIOS[req.scenario_id]
    agents = population(max(50, min(req.population, 2000)), req.seed)
    res = swarm.simulate(agents, sc["products"], req.target, req.promo, seed=req.seed)
    res["scenario_name"] = sc["name"]
    res["products"] = sc["products"]
    return res


@app.get("/")
def index():
    return FileResponse(BASE / "templates" / "index.html")


@app.get("/api/bootstrap")
def bootstrap():
    return {
        "scenarios": SHELF["scenarios"],
        "promo_types": {k: v["label"] for k, v in PROMO_TYPES.items()},
        "segments": segment_table(),
    }


@app.post("/api/simulate")
def simulate(req: SimReq):
    return JSONResponse(_run(req))


@app.post("/api/narrate")
def narrate_ep(req: SimReq):
    res = _run(req)
    return narrate(res, res["scenario_name"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
