"""The only place FastAPI() is instantiated for this project. Mounts
webhook.router; kept beside the router it wires up rather than a new
src/api/ package, since nothing else in the planned dependency graph adds
a second HTTP router.

Loads .env itself: unlike pytest (tests/conftest.py loads it once for the
whole suite), `uvicorn src.ingest.app:app` is this process's only entry
point, so if it doesn't load .env, nothing does -- DATABASE_URL and
RAZORPAY_WEBHOOK_SECRET would silently be missing. Matches the same
find_dotenv(usecwd=True) pattern run.ps1's verify probe already uses.
"""
from __future__ import annotations

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

from fastapi import FastAPI  # noqa: E402

from src.ingest import webhook  # noqa: E402

app = FastAPI(title="Mandate Recovery Engine -- ingest")
app.include_router(webhook.router)
