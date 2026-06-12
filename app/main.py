"""FastAPI application entrypoint."""
from __future__ import annotations

import asyncio
import pathlib
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from app.api.routes import router
from app.core.textproc import IngressRejection
from app import deps

_DASHBOARD = pathlib.Path(__file__).resolve().parent.parent / "dashboard" / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.to_thread(deps.get_db)   # create schema on boot
    yield


app = FastAPI(
    title="Parallax",
    description="Converts one biased query into a verified multi-perspective corpus: "
                "detects and neutralizes the query's presupposition, fans out "
                "epsilon-constrained perspective searches (including dialectic "
                "affirm/counter probes), and proves diversity and premise-neutrality "
                "with live OLAP metrics computed in ClickHouse SQL.",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

app.include_router(router)


@app.exception_handler(IngressRejection)
async def ingress_handler(request, exc):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.get("/", include_in_schema=False)
async def dashboard():
    return FileResponse(_DASHBOARD)
