#!/usr/bin/env python3
"""
Run the app locally with Sideline Pro on and PayPal faked in memory, for
working on the coach and ops screens before sandbox keys exist.

    python tools/dev_server_fake_paypal.py            # http://127.0.0.1:8010

What is real: the database (a SQLite file under .data/dev-fake/), the review
state machine, the sweeper, the coach draft (the template, or Nemotron when
NEBIUS_API_KEY is set), the ops agent (Nemotron; needs NEBIUS_API_KEY).
What is faked: every PayPal call (tests/pro_helpers.FakePayPal) and the
Agent Toolkit bridge. The page loads PayPal's JS SDK with the generic
sandbox client id `sb`, so the button renders, but its popup cannot approve
a fake order — approve from the command line instead:

    curl -s -X POST localhost:8010/api/checkout/reviews -H 'content-type: application/json' \\
      -d '{"source":"sample","athlete_token":"dev","athlete_name":"Dan"}'
    curl -s -X POST localhost:8010/api/checkout/reviews/<review_id>/approve

Then switch to the Coach tab. Never deploy this: it is a development runner.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

os.environ.setdefault("PAYPAL_ENV", "sandbox")
os.environ["PAYPAL_CLIENT_ID"] = "sb"  # PayPal's generic sandbox id: the button renders
os.environ["PAYPAL_CLIENT_SECRET"] = "fake-local-secret"
os.environ.setdefault("PRO_DB_PATH", str(ROOT / ".data" / "dev-fake" / "pro.sqlite"))
os.environ.setdefault("DEMO_SAMPLE_DIR", str(ROOT / "demo_sample"))
# The coach/ops sign-in for this runner only. Never the deploy's password.
os.environ.setdefault("COACH_PASSWORD", "dev")
os.environ.pop("DATABASE_URL", None)

from server.services import ops_agent  # noqa: E402
from server.services.paypal_client import reset_paypal_for_tests  # noqa: E402
from tests.pro_helpers import FakePayPal, FakeToolkit  # noqa: E402

fake = FakePayPal()
reset_paypal_for_tests(fake)
ops_agent.reset_toolkit_for_tests(FakeToolkit(fake))

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("DEV_PORT") or "8010")
    print(f"Sideline Pro dev server with a FAKE PayPal on http://127.0.0.1:{port}  (Ctrl-C to stop)")
    print(f"Coach/ops sign-in password: {os.environ['COACH_PASSWORD']}  (COACH_PASSWORD)")
    uvicorn.run("server.main:app", host="127.0.0.1", port=port, reload=False, log_level="info")
