"""
ShelfMind - FastAPI backend (retailer-choice model).

  POST /api/simulate -> run the swarm (fast, deterministic maths)
  POST /api/narrate  -> re-run the same scenario and add the LLM explanation

The narrate endpoint recomputes from the same (product, promo, population, seed)
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
from personas import generate_population, retailer_table, segment_table

BASE = pathlib.Path(__file__).parent
PRODUCTS = json.loads((BASE / "data" / "products.json").read_text(encoding="utf-8"))["products"]
BY_ID = {p["id"]: p for p in PRODUCTS}

app = FastAPI(title="ShelfMind")
_pop_cache: dict[tuple[int, int], list] = {}


def population(n: int, seed: int):
    key = (n, seed)
    if key not in _pop_cache:
        _pop_cache[key] = generate_population(n, seed)
    return _pop_cache[key]


class SimReq(BaseModel):
    product_id: str
    promo: dict               # {"type": ..., "depth": ...}
    population: int = 1000
    seed: int = 7


def _run(req: SimReq) -> dict:
    product = BY_ID[req.product_id]
    agents = population(max(50, min(req.population, 2000)), req.seed)
    return swarm.simulate(product, agents, req.promo, seed=req.seed)


@app.get("/")
def index():
    return FileResponse(BASE / "templates" / "index.html")


@app.get("/api/bootstrap")
def bootstrap():
    return {
        "products": [{"id": p["id"], "name": p["name"], "category": p["category"],
                      "tesco_base": p["tesco_base"]} for p in PRODUCTS],
        "promo_types": {k: v["label"] for k, v in PROMO_TYPES.items() if k != "none"},
        "segments": segment_table(),
        "retailers": retailer_table(),
    }


@app.post("/api/simulate")
def simulate(req: SimReq):
    return JSONResponse(_run(req))


@app.post("/api/narrate")
def narrate_ep(req: SimReq):
    res = _run(req)
    return narrate(res)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
