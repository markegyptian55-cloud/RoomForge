from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError

from .config import get_settings
from .db import Base, engine
from .routers import auth, edits, projects, scenes, tasks
from .services.queue import ConflictError

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
settings = get_settings()
Base.metadata.create_all(engine)
app = FastAPI(title="Room3D API")
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins,
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    return response


@app.exception_handler(LookupError)
async def lookup_handler(_: Request, exc: LookupError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"error": "not_found", "detail": str(exc)})


@app.exception_handler(ConflictError)
async def conflict_handler(_: Request, exc: ConflictError) -> JSONResponse:
    return JSONResponse(status_code=409, content={"error": "conflict", "detail": str(exc)})


@app.exception_handler(IntegrityError)
async def integrity_handler(_: Request, __: IntegrityError) -> JSONResponse:
    return JSONResponse(status_code=409, content={"error": "conflict", "detail": "resource conflict"})


@app.exception_handler(Exception)
async def error_handler(_: Request, exc: Exception) -> JSONResponse:
    LOGGER.exception("Unhandled server error: %s", exc)
    return JSONResponse(status_code=500, content={"error": "internal_error", "detail": "internal server error"})


app.include_router(auth.router)
app.include_router(projects.router)
app.include_router(scenes.router)
app.include_router(tasks.router)
app.include_router(edits.router)
