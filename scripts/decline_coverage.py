"""Prints decline_class coverage from `ingested_event` -- specifically the
UNKNOWN rate, which payments-domain's B3 review flagged as "a reported
metric" that nothing yet reports (PLAN_DETAIL.md's B3 section pre-concedes
low taxonomy coverage on the condition that its rate is visible, not
swallowed).

Deliberately NOT wired into scripts/show_state.py (the SessionStart hook):
that script must stay fast and must not fail a session start just because
Docker/Postgres happens to be down, and it has no DB dependency today. This
is a separate, on-demand check instead -- run it whenever the ingest
pipeline has real rows to look at.

Usage:
    python scripts\\decline_coverage.py
"""
from __future__ import annotations

import sys

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

from src.core.db import connect  # noqa: E402


def main() -> int:
    try:
        conn = connect()
    except Exception as exc:
        print(f"Postgres unreachable: {exc}")
        return 1

    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM ingested_event")
        total = cur.fetchone()[0]

        if total == 0:
            print("ingested_event is empty -- nothing to report yet.")
            conn.close()
            return 0

        cur.execute(
            "SELECT decline_class, count(*) FROM ingested_event "
            "GROUP BY decline_class ORDER BY count(*) DESC"
        )
        rows = cur.fetchall()

        cur.execute(
            "SELECT count(*) FROM ingested_event WHERE decline_class = 'UNKNOWN'"
        )
        unknown = cur.fetchone()[0]

    conn.close()

    print(f"ingested_event: {total} row(s)\n")
    for decline_class, count in rows:
        label = decline_class if decline_class is not None else "(null)"
        pct = 100.0 * count / total
        print(f"  {label:<20} {count:>6}  ({pct:5.1f}%)")

    unknown_pct = 100.0 * unknown / total
    print(f"\nUNKNOWN rate: {unknown_pct:.1f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
