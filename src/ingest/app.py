"""The only place FastAPI() is instantiated for this project. Mounts two
routers: `src/ingest/webhook.py` (events arriving) and `src/api/read.py`
(reads going out).

This docstring previously read "kept beside the router it wires up rather
than a new src/api/ package, since nothing else in the planned dependency
graph adds a second HTTP router." R6 added one (reports/gates.md,
"Post-B16 remediation gates"), so that sentence is REWRITTEN rather than
left standing next to the code disproving it. `src/api/` is a separate
package deliberately: ingest means events arriving from the outside, and
these three endpoints are reads going out -- a different direction, a
different failure mode (a read cannot lose a write), and a different
review surface.

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

from src.api import read  # noqa: E402
from src.ingest import webhook  # noqa: E402

app = FastAPI(title="Mandate Recovery Engine")
app.include_router(webhook.router)
app.include_router(read.router)
