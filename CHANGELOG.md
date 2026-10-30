# Changelog

## Unreleased — Sideline Pro (PayPal AI Hackathon), branch `paypal`

The paid human-coach review on top of the judge, in the PayPal **sandbox**.
Everything below is new since the `v1-nebius` tag (Oct 30, 2026); nothing
about scoring, prompts or the rubric changed. See
[What's new for PayPal](README.md#sideline-pro--whats-new-for-the-paypal-ai-hackathon-since-oct-30)
in the README for the design.

### Added
- **Checkout on every verdict card** (real results and the cached sample): "Get a coach's eyes —
  $15", a PayPal JS SDK button loaded with `intent=authorize`, the held-not-charged line, and a
  status page per review at `/review/{id}`.
- **PaymentService** (`server/services/paypal_client.py`): OAuth client-credentials token (cached),
  Orders v2 create (intent AUTHORIZE, `custom_id` = review id) and authorize, Payments v2 capture,
  void and refund, order lookup, webhook signature verification, Invoicing v2 helpers. Every write
  sends a `PayPal-Request-Id`. The sandbox base URL is forced; `PAYPAL_ENV=live` refuses to start.
- **Durable store** (`server/services/db.py`, SQLAlchemy): Heroku Postgres via `DATABASE_URL`, a
  SQLite file otherwise. Tables `athlete`, `coach`, `lift_snapshot` (+ JPEG keyframes),
  `review_order`, `payment_event`, `agent_action`, `invoice_ref`.
- **Lift snapshot at purchase** (`server/services/snapshots.py`): the judgment JSON, the
  measurement, the clip warnings and up to four evidence stills are copied into the database when
  the athlete taps Pay, so a review outlives the 30-minute job and a dyno restart.
- **Review state machine** (`server/services/reviews.py`): CHECKOUT_CREATED → AUTHORIZED →
  IN_REVIEW → CAPTURED | VOIDED (sla_expired / coach_declined / agent_cancel / capture_failed) |
  REFUNDED; ABANDONED for checkouts never approved. Every transition is a compare-and-set on the
  status column (the coach and the sweeper cannot race), no capture after `sla_due_at`, one retry
  on a transient capture error then the hold is released, every PayPal call leaves a
  `payment_event`.
- **SLA sweeper** (`server/services/sweeper.py`): every minute in-process plus on every queue or
  status request, so a dyno that slept through a deadline still releases the hold. `SLA_HOURS`
  defaults to 48 and is capped at 72 (PayPal's capture honor period).
- **Coach queue and editor** (Coach tab): held reviews nearest deadline first with a countdown;
  keyframes, the AI report, the **AI draft** (clearly labelled, editable), score/band override,
  Deliver (capture) and Decline (void).
- **AI coach draft** (`server/services/coach_draft.py`, `prompts/coach_draft.txt`): one Nemotron
  call per paid review, made once when the hold is placed, validated against a fixed schema
  (summary, 1–3 cues with evidence refs, drill, confidence note, suggested score/band); a
  deterministic template from the judge's cues when the model fails or `DRAFT_DAILY_LIMIT` is
  spent. The athlete only ever sees what the coach delivered, marked "Reviewed by Coach Maya
  (AI-assisted)".
- **Ops agent** (`server/services/ops_agent.py`, `prompts/ops_agent.txt`, Ops tab): OpenAI-style
  function calling on Nebius Token Factory, at most 6 tool rounds. App tools `find_reviews`,
  `list_athletes`, `earnings_summary` (local ledger), `cancel_or_refund_review`,
  `invoice_athletes`; PayPal Agent Toolkit tools `list_invoices`, `get_invoice`,
  `get_order_details`, `send_invoice_reminder`, `cancel_sent_invoice`, and `create_invoice` +
  `send_invoice` behind `invoice_athletes`. **Writes only propose**: a card with the exact effect,
  amount and recipient; a tap on Confirm (single-use token, 10 minutes) re-validates and executes.
  Audit log of every action. Allowlist in code: no `create_order`/`pay_order`, subscriptions,
  products, disputes, shipments or `get_merchant_insights`.
- **Guardrails in code**: sandbox-only assertion; one release/refund per review for its own
  amount; invoices ≤ `PAYPAL_INVOICE_MAX_USD` (200) each and ≤ `AGENT_INVOICE_MAX_COUNT` (5) per
  request, USD, recipients from `list_athletes` only; idempotency keys; `AGENT_DAILY_LIMIT`,
  `AGENT_HOURLY_LIMIT` and `DRAFT_DAILY_LIMIT` budgets separate from the judge's; prompts and
  names passed as quoted data; tool output summarised, never followed.
- **Role switcher** (Athlete / Coach / Ops, no reload; `#coach`, `#ops` and `#coach:<id>` deep
  links) and a **SANDBOX · no real money** banner on every screen.
- **Demo-only controls** (`PRO_DEMO_MODE`, optional `ADMIN_DEMO_KEY`): "Expire now" on any held
  review (the sweeper then voids it) and "Reset demo data" (releases every hold, re-seeds Coach
  Maya and the athletes Dan, Noa and Lior).
- **PayPal webhook** (`POST /paypal/webhook`, P1): signature verified with PayPal, idempotent by
  event id; `PAYMENT.AUTHORIZATION.VOIDED`, `PAYMENT.CAPTURE.COMPLETED/REFUNDED/DECLINED`,
  `INVOICING.INVOICE.PAID` keep the rows honest. Off until `PAYPAL_WEBHOOK_ID` is set.
- **Coach-time metric**: review time (draft opened → delivered) and the share of the AI draft's
  text kept, stored with each delivered review.
- `tools/secret_scan.py` (+ a test that runs it over the tree) so no key is ever committed;
  `tools/dev_server_fake_paypal.py` to work on the screens with PayPal faked in memory.
- `bin/post_compile` (Heroku Python buildpack hook) and a Dockerfile step that install
  `paypal-agent-toolkit==1.11.0` with `--no-deps`; `app.json` declares the Postgres add-on.
- Tests: `tests/test_pro_api.py`, `tests/test_ops_agent.py`, `tests/test_coach_draft.py`,
  `tests/test_secret_scan.py` with a fake sandbox PayPal, a fake toolkit bridge and a scripted
  model. No network anywhere.

### Changed
- `requirements.txt`: `sqlalchemy`, `psycopg[binary]`.
- `/api/config` carries a public `pro` block (enabled, env, the public client id, price, SLA).
- `README.md`, `.env.example`, `deploy/heroku.sh`, `deploy/heroku-web.sh`, `Dockerfile`,
  `app.json` document and set the Pro variables.

## v1-nebius — 2026-10-30

Sideline Lift Judge as submitted to the Nebius × NVIDIA Global AI Hackathon: the plate tracker,
MiniCPM-V observations, the Nemotron judge, evidence-backed cues, the annotated replay, the open
public demo with spend caps and the cached sample. Tagged before any PayPal work.
