## Sideline Lift Judge

Film one Olympic lift from the side on your phone and get back a **1–10 bar-path efficiency
score**, a band, the **maximum horizontal bar drift**, a **confidence** value, notes and **1–2
coaching cues** — plus an annotated replay you can download or share. Mobile-first, so a coach
can use it from the side of the platform.

▶ **Demo video:** <https://www.youtube.com/watch?v=3E22yknw8Nk>

**New — Sideline Pro:** a paid human-coach review on top of the judge. **PayPal** (sandbox):
Orders v2 authorize-then-capture as escrow, Payments v2 capture / void / refund, Invoicing v2
through the **PayPal Agent Toolkit**. **AI** (Nebius Token Factory, NVIDIA Nemotron): the judge,
an AI-drafted coach review, and an ops agent with function calling that only ever *proposes* a
payment action until a human taps Confirm. See
[What's new for the PayPal AI Hackathon](#sideline-pro--whats-new-for-the-paypal-ai-hackathon-since-oct-30).

Built for the Nebius × NVIDIA Global AI Hackathon. The judge is an **NVIDIA Nemotron 3** model —
**Super** in the public demo, Ultra by default in a checkout — and both stages run on
**Nebius Token Factory**.

**Demo:** <https://sideline-lift-judge-99d464f34df0.herokuapp.com/> — open it on a phone.
**No code needed**: the sample result is one tap away, your own clip two. The live
judge is `nvidia/nemotron-3-super-120b-a12b` (no Ultra fallback) so the bill stays
bounded; a local checkout defaults to **Ultra (`nvidia/Nemotron-3-Ultra-550b-a55b`)**
with a Super fallback. The sample result is served from a Google Drive-hosted clip
the dyno downloads once (see
[Running the public demo](#running-the-public-demo-without-a-surprise-bill) for
the `SAMPLE_CLIP_URL` env var). The demo has a daily limit and its counters reset
when the dyno restarts (see [Limitations](#limitations)).\
▶ **Demo video:** <https://www.youtube.com/watch?v=3E22yknw8Nk>

- Mobile-first web app, installable as a PWA — film or upload, trim, score, share
- Two kinds of evidence, one judge: a local plate tracker **measures** the bar path and a vision
  model **describes** it; NVIDIA Nemotron weighs both against the rubric
- Clean & jerk is detected and scored as **two separate lifts**
- Confidence drops, with a visible warning, when the clip's own measurements are poor: the
  bar leaves the frame, the plate is too small to track accurately, the camera is off-axis, the
  tracker lost the plate, the window held more than one lift, or the judge's verdict contradicts
  the measured path. Portrait by itself is not a problem — most phones film this way and a
  well-framed portrait clip scores at full confidence
- Every verdict card says **which model did what** — the local tracker measured, MiniCPM-V
  described, which Nemotron judged (and whether the fallback answered) — and its drift line is
  written from the measurement, so the words cannot contradict the number
- Schema-validated JSON, a repair retry, and a fallback judge model
- Two CLIs for one clip and for a whole session
- An **open public demo** bounded on the server, not by a code: budgets counted in model calls
  (per day, per address, with a slice of the day kept for newcomers), a cost estimate before the
  first call, a run deadline, a cheaper judge with no fallback, a kill switch, and a free cached
  sample so a judge never meets a dead end — all on by default in the deploys, and the access
  mode (`open` / `link` / `code`) is one config variable

## Sideline Pro — what's new for the PayPal AI Hackathon (since Oct 30)

> **An AI judge that scores your lift in a minute. A human coach confirms it in 48 hours, or you
> never pay.**

Sideline Pro adds a **paid human-coach review** to the judge above, built for the
[PayPal AI Hackathon](https://paypalaihackathon.devpost.com/). The athlete gets the AI bar-path
score as before, then buys a coach review through **PayPal Checkout**. The money is **authorized
at purchase, captured only when the coach delivers, and voided automatically if the coach misses
the 48 h deadline** — authorize-then-capture as escrow. The coach's queue opens every review with
the **AI report already written up as a draft**, and a plain-English **ops agent** handles
releases, refunds and invoices with a human tap between the model and the money: releases and
refunds are the app's own PayPal Payments v2 calls, invoices go through the **PayPal Agent
Toolkit**. Everything runs in the **PayPal sandbox**; this build refuses a live environment. The state of
the repo before any of this is tagged [`v1-nebius`](https://github.com/amitmtrn/sideline-lift-judge/releases/tag/v1-nebius);
[CHANGELOG.md](CHANGELOG.md) lists every change.

### The loop

```
 athlete                      PayPal (sandbox)                     coach                       ops agent
 ───────                      ────────────────                     ─────                       ─────────
 score a lift (or the sample)
 tap "Get a coach's eyes — $15"
   ├─ POST /api/checkout/reviews ──► Orders v2: create (AUTHORIZE,
   │   lift snapshotted into Postgres    custom_id = review id)
   ├─ PayPal button (JS SDK,
   │   intent=authorize) ─ approve ─► Orders v2: authorize ──► AUTHORIZED: $15 HELD, sla_due_at = +48 h
   │                                                            AI draft written (1 Nemotron call)
   │                                                          queue, nearest deadline first
   │                                                          open → IN_REVIEW, edit the draft
   │                                   Payments v2: capture ◄── Deliver ──► CAPTURED: charged $15
   │                                   Payments v2: void    ◄── Decline / SLA sweeper ──► VOIDED: not charged
   └─ /review/{id}: "Held" → "Delivered · charged" or "Released · not charged"
                                       Payments v2: refund  ◄── "refund Dan" → card → Confirm ──┘
                                       Invoicing v2 (Agent Toolkit) ◄── "invoice my 3 athletes $60" → card → Confirm
```

**The state machine** (`server/services/reviews.py`), one row per review, every transition a
compare-and-set on the status column:

```
CHECKOUT_CREATED ──approve──► AUTHORIZED ──coach opens──► IN_REVIEW ──Deliver (capture)──► CAPTURED ──refund──► REFUNDED
       │                          │                           │
       │ 30 min, never approved   └────── SLA sweeper (48 h) · Decline · agent release · capture failed ──► VOIDED
       ▼
   ABANDONED
```

`CAPTURING`, `VOIDING` and `REFUNDING` are the transient claims while PayPal is on the line; a
claim older than five minutes (a crashed worker) is reconciled against `GET /v2/checkout/orders`.

**Who may do what.** Athletes need no account: a review is reached by its unguessable id
(`/review/<128 random bits>`) plus the anonymous token the page keeps. The **Coach** and **Ops**
tabs — queue, deliver, decline, the agent, Expire now, Reset — are behind one password
(`COACH_PASSWORD`, a signed HttpOnly session cookie, checked on every route server side, not just
hidden in the UI). Judges get that password in the Devpost testing instructions only.

### What changed since `v1-nebius`

- **Checkout on every verdict card**, the cached sample included (so judges can run the whole
  payment loop at zero model cost): a PayPal JS SDK button loaded with `intent=authorize`, the
  line "Held, not charged, until Coach Maya delivers (48 h)", and a status page per review.
- **PaymentService** (`server/services/paypal_client.py`): the only place a PayPal URL is
  spelled. OAuth client-credentials token (cached), Orders v2 create + authorize, Payments v2
  capture / void / refund, order lookup, webhook verification, Invoicing v2 helpers. Every write
  carries a `PayPal-Request-Id`, so a retry cannot double-charge or double-void.
- **A durable store** (`server/services/db.py`): Heroku Postgres (`DATABASE_URL`), SQLite in a
  checkout. The lift — judgment JSON, measurement, warnings and up to four evidence stills — is
  **snapshotted at purchase**, because a job lives thirty minutes and a dyno disk is wiped on
  restart.
- **A review state machine with an SLA sweeper** (`server/services/reviews.py`,
  `server/services/sweeper.py`), compare-and-set on every transition, no capture after the
  deadline, one retry on a transient capture error then the hold is released.
- **Coach queue + editor** with the **AI draft** (`server/services/coach_draft.py`): one
  Nemotron call per paid review, schema-validated, a template when the model or the budget
  fails. Deliver captures; Decline voids; a coach may override the score and band.
- **The ops agent** (`server/services/ops_agent.py`): function calling on Nebius Token Factory,
  PayPal Agent Toolkit tools plus app tools, **confirm-before-write cards**, an audit log, and
  limits enforced in code.
- **Role switcher** (Athlete / Coach / Ops, no reload), a **SANDBOX · no real money** banner on
  every screen, the **coach/ops sign-in** (`COACH_PASSWORD`) in front of every coach and ops
  route, and demo-only **Expire now** and **Reset demo data** buttons for signed-in staff.
- A PayPal **webhook** endpoint (P1), a **secret scan** that runs in the tests, a dev runner with
  PayPal faked in memory, and ~70 new tests with no network anywhere.

### PayPal integration

| Step | API | Endpoint | Notes |
| --- | --- | --- | --- |
| Button | JavaScript SDK | `/sdk/js?client-id=…&intent=authorize` | `intent=authorize`, or the SDK defaults to capture |
| Create | Orders v2 | `POST /v2/checkout/orders` | `intent: AUTHORIZE`, `custom_id` = review id, $15 USD |
| Hold | Orders v2 | `POST /v2/checkout/orders/{id}/authorize` | in `onApprove`, server side; stores the authorization id and expiry |
| Deliver | Payments v2 | `POST /v2/payments/authorizations/{id}/capture` | `final_capture: true`; stores the capture id |
| Release | Payments v2 | `POST /v2/payments/authorizations/{id}/void` | SLA miss, Decline, agent cancel, capture failure |
| Refund | Payments v2 | `POST /v2/payments/captures/{id}/refund` | full amount only |
| Reconcile | Orders v2 | `GET /v2/checkout/orders/{id}` | after a crash mid-transition, and on `ORDER_ALREADY_AUTHORIZED` |
| Invoices | Invoicing v2 via the **Agent Toolkit** | `create_invoice`, `send_invoice`, `list_invoices`, `get_invoice`, `send_invoice_reminder`, `cancel_sent_invoice` | to the seeded athletes' sandbox personal accounts |
| Webhook (P1) | Notifications | `POST /v1/notifications/verify-webhook-signature` | `/paypal/webhook`; simulator events cannot verify, use real sandbox payments |

**Timing that shaped the design.** An authorization holds funds for 29 days but capture is only
guaranteed inside the **3-day honor period**; the 48 h SLA fits inside it, so there is no
reauthorize logic and `SLA_HOURS` is capped at 72 in code. `PAYMENT.AUTHORIZATION.VOIDED` also
fires when an authorization expires; the webhook handles it idempotently.

**The PayPal Agent Toolkit (Python), honestly.** `paypal-agent-toolkit` 1.11.0 does not install on
Python 3.13 as published — its pins (`openai-agents==0.0.2`, `crewai-tools`, `langchain`) do not
resolve — so the Dockerfile and `bin/post_compile` (the Heroku buildpack hook) run
`pip install --no-deps paypal-agent-toolkit==1.11.0` and the app uses the framework-free
`paypal_agent_toolkit.shared` layer: its tool schemas become OpenAI-format tool definitions and
`PayPalAPI.run(method, params)` executes them. The Python toolkit has **no refund tool and no
authorization capture/void tools** (only the TypeScript one has `create_refund`), so release and
refund are **app tools** that wrap the PaymentService and keep the state machine authoritative;
the toolkit does invoices and order lookup. `get_merchant_insights` is not exposed (it fails in
sandbox) and earnings come from the local ledger rather than Transaction Search, which lags up to
three hours. If the package is missing the invoice tools fall back to the built-in Invoicing v2
client; the Ops tab and `/api/ops/overview` say which one is in use.

### AI, in three places

1. **The judge**, unchanged: OpenCV plate tracker → MiniCPM-V 4.5 observations → NVIDIA
   Nemotron 3 Super verdict with evidence-backed cues (everything above this section).
2. **The coach draft** (`prompts/coach_draft.txt`): the judgment JSON, the measurement, the
   warnings and the athlete's note (quoted data) go to Nemotron once, when the hold is placed.
   The reply must validate as `{summary, cues[1–3]{text, evidence_ref}, drill, confidence_note,
   suggested_score, suggested_band}`; otherwise, or when `DRAFT_DAILY_LIMIT` is spent, a
   deterministic template built from the judge's cues takes its place. The editor labels it
   "AI draft — edit before delivering"; the athlete sees only what the coach delivered, marked
   "Reviewed by Coach Maya (AI-assisted)". The delivered review records how long it took and how
   much of the draft's text survived.
3. **The ops agent** (`prompts/ops_agent.txt`): system prompt + the request (quoted) → model with
   tools → at most 6 tool rounds → an answer that lists the tools it called. `AGENT_MODEL`
   defaults to the judge model; set another Token Factory model with solid function calling if
   Nemotron's tool calls prove unreliable.

**Guardrails, in code rather than the prompt:**
- Write tools never execute inside the model loop. The call becomes an `agent_action` row in
  status `PROPOSED` and a card with the exact effect, amount and recipient. Only a tap on
  Confirm — a single-use token, valid ten minutes — executes it, and the server re-validates the
  review's state and the limits at that moment.
- Allowlist: `create_order`, `pay_order`, subscriptions, products, disputes, shipments and
  `get_merchant_insights` are not on the menu, in the schema list or at execution.
- One release/refund per review, for exactly its amount. Invoices at most
  `PAYPAL_INVOICE_MAX_USD` (200) each and `AGENT_INVOICE_MAX_COUNT` (5) per request, USD, to
  athletes `list_athletes` knows. Sandbox-only assertion before every call. Idempotency keys.
- Budgets of their own: `AGENT_DAILY_LIMIT` model calls per day, `AGENT_HOURLY_LIMIT` requests
  per visitor per hour, `DRAFT_DAILY_LIMIT` drafts per day — separate from the judge's
  `DEMO_DAILY_LIMIT`, so the payment loop on the sample never competes with clip scoring.
- Names, notes and invoice text are quoted data; tool output is summarised, never followed.

### Running it (sandbox)

1. **PayPal**: developer.paypal.com → Apps & Credentials (**Sandbox**) → create a REST app
   linked to the default sandbox *business* account (that account is "Coach Maya", the
   merchant). Note the client id and secret. Use the default sandbox *personal* account as the
   buyer, or create one under Sandbox → Accounts. Optional: put the sandbox personal emails of
   three test accounts in `DEMO_ATHLETE_EMAILS` so the agent's invoices land in the sandbox.
2. **Config vars** (names in `.env.example`; values never in the repo):
   `PAYPAL_ENV=sandbox`, `PAYPAL_CLIENT_ID`, `PAYPAL_CLIENT_SECRET`, **`COACH_PASSWORD`** (the
   Coach and Ops tabs stay locked without it), and on Heroku `DATABASE_URL` from the Postgres
   add-on. Optional: `SESSION_SECRET`, `STAFF_LOGIN_HOURLY_LIMIT`, `PAYPAL_WEBHOOK_ID`,
   `SLA_HOURS` (48; never above 72), `REVIEW_PRICE_USD`, `COACH_NAME`, `COACH_EMAIL`,
   `DEMO_ATHLETE_EMAILS`, `AGENT_MODEL`, `DRAFT_DAILY_LIMIT`, `AGENT_DAILY_LIMIT`,
   `AGENT_HOURLY_LIMIT`, `PRO_DEMO_MODE`. Pro is on when the client id and secret are both set;
   without them the judge runs exactly as before and the paid flow stays hidden.
3. **Heroku** (buildpack path, what the live app runs):
   ```bash
   heroku addons:create heroku-postgresql:essential-0 -a <app>
   heroku config:set PAYPAL_ENV=sandbox PAYPAL_CLIENT_ID=… PAYPAL_CLIENT_SECRET=… COACH_PASSWORD=… -a <app>
   git push heroku paypal:main          # or Deploy Branch in the dashboard
   heroku run python -c "from server.services.paypal_client import get_paypal; print(get_paypal().access_token()[:8], 'ok')" -a <app>
   ```
   `bin/post_compile` installs the Agent Toolkit during the build (its log line says
   `toolkit tools: 42`). The startup log prints one `Pro (coach review): on …` line naming the
   store; a SQLite store on a dyno is warned about, since it would not survive a restart.
   `deploy/heroku.sh` (container path) passes the same variables through and creates the add-on.
4. **Locally, with real sandbox keys**: put them in `.env` and `uvicorn server.main:app --reload`.
   The store is `.data/sideline-pro.sqlite`. **Without keys**, work on the screens with PayPal
   faked in memory: `python tools/dev_server_fake_paypal.py` (port 8010; its docstring shows how
   to approve a review from the command line, since a fake order cannot pass PayPal's popup).
5. **Sandbox walkthrough** (the smoke test after every deploy): `/api/config` →
   `pro.pro_enabled: true` and `pro.staff_login_configured: true`. Open the sample result → *Get a
   coach's eyes* → pay with a **US, English** sandbox buyer (the PayPal popup is in English; the
   SDK is loaded with `locale=en_US` as a fallback) → the athlete page shows **Held** with the
   order and authorization ids. **Coach** tab → sign in → open the review → edit the AI draft →
   *Deliver* → the capture id shows and the athlete page says **Delivered · charged**. Buy a
   second review → **Ops** → *Expire now* → that athlete sees **Released · not charged**. Ops
   agent: "Refund <name>'s review, she disputed it" → card → *Confirm* → the refund id is in the
   audit log and the reply renders as clean text. "Invoice my 3 athletes $60 each" → *Confirm* →
   three invoices reach **SENT** with their ids in the card and the audit log; a PayPal refusal
   reads as its error name, the issue and the `debug_id`, never as a bare 500. Signed out,
   `curl` each coach/ops route → **401**. `python tools/secret_scan.py` → clean.

### For the judges (testing instructions)

The hosted demo stays free and unrestricted through **Dec 15, 2026 (PT)**. The sandbox buyer
login and the coach/ops password go **only** into the Devpost testing-instructions field — never
this README, the video or the repo:

> Sandbox only — no real money. Pay with the PayPal sandbox buyer (US, English): email
> `<SANDBOX_BUYER_EMAIL — in Devpost only>` / password `<in Devpost only>`.
> The **Coach** and **Ops** tabs ask for a password: `<COACH_PASSWORD — in Devpost only>`.
> Steps: open the sample result → *Get a coach's eyes* → pay with the sandbox buyer → switch to
> the **Coach** tab → edit the AI draft → *Deliver* (captured) → **Ops** tab: "refund the review
> for <athlete>" → *Confirm*. To see the 48 h auto-release: **Ops** → *Expire now* on any held
> review. **Ops** → *Reset demo data* puts everything back (holds released, athletes re-seeded).

The whole payment loop runs on the cached sample, so the judge's model budget is never a
blocker; the ops agent works on the seeded demo athletes (Dan, Noa, Lior).

### Tests for the Pro loop

`tests/test_pro_api.py` (checkout → hold → queue → draft → deliver/capture; decline; the SLA
sweep refusing a late capture; retry then release on capture failure; a review surviving a
restart; what each role may see; every coach and ops route answering 401 without the sign-in,
the cookie's signature and expiry, wrong passwords rate-limited; the lift stored as a key and
shown as a word), `tests/test_ops_agent.py` (writes only propose; confirm executes;
wrong/expired/second tokens refused; re-validation at confirm time; three invoices as one card
through the toolkit bridge, created → sent → read back; a PayPal refusal on an invoice surfaced
with its name, issue and `debug_id` as a 502, one bad line not sinking a batch, a crash recorded
as FAILED; every limit; the denied tools; budgets; the loop's step cap),
`tests/test_coach_draft.py` (valid JSON kept, prose tolerated, invalid JSON and model errors fall
back to the template, the draft budget) and `tests/test_secret_scan.py`. PayPal, the toolkit and
the model are all faked in `tests/pro_helpers.py`; nothing touches the network.

### How a clip becomes a verdict

```
 phone clip ──► trim ──► tap the near plate (optional) ──► plate tracker (OpenCV, local, free)
                            │  finds each lift: where the plate rises; a rise that starts
                            │  back on the floor is the next rep
                            │  follows the near plate through it — side-on, or as an
                            │  ellipse from an angled camera — and checks it IS a plate
                            ├─► bar-path MEASUREMENT  (drift in plate diameters,
                            │   where in the rise, coverage)          ──────────┐
                            ▼                                                   │
                      8–16 frames from the lift window                          │
                            │                                                   │
                            ▼                                                   ▼
             Stage 1  MiniCPM-V (Token Factory)              Stage 2  NVIDIA Nemotron 3
             "what the bar did", phase by phase   ────────►  rubric + observations + measurement
             direction, bar–body contact, angle              → validated JSON: score, band, drift,
                                                               cues, confidence
                                                                        │
                   confidence caps (local): plate out of frame, plate   │
                   too small, off-axis, mixed lifts, no trace,           │
                   verdict vs. measurement  ◄──────────────────────────┐ │
                                                                       │ │
                                                                        │
                                                                        ▼
                   annotated replay (ffmpeg): verdict strip + the same tracked path drawn on it
```

The measurement and the trace on the replay are one and the same numbers, so what the judge says
about the drift and what the orange ring shows cannot disagree by accident — and when the judge
ignores the number, a local re-check says so and caps the confidence.

### Stack
- FastAPI + Jinja2, mobile-first templates, web app manifest and a service worker
- OpenCV for the motion scan and frame extraction; ffmpeg for trimming and the annotated replay
- OpenAI-compatible HTTP calls to Nebius Token Factory
- Pydantic for JSON schema validation
- Pytest throughout, with every Token Factory call mocked

### Project layout
Server and client are separate top-level folders. Nothing in `client/` is imported as Python —
the server only reads it off disk and serves it — so the front end can be replaced (or a native
app added, see the roadmap) without touching the pipeline.

```
server/              the Python app — nothing here is sent to the browser
  main.py            FastAPI routes: pages, PWA assets, jobs API, replay files
  services/          pipeline.py (a scoring run), nebius_client.py (Token Factory),
                     measurement.py (the tracker's numbers, for the judge and the API),
                     confidence.py (FR-8 caps and the verdict re-check), replay.py
                     (ffmpeg overlay), jobs.py, errors.py (plain messages for users),
                     results.py (JSON shape), demo_guard.py + sample.py (public-demo caps)
  schemas/           Pydantic models — the judge's JSON contract
  utils/             bar_track.py (plate tracker: finds lifts, the legacy trace),
                     plate_follow.py (the drawn trace: optical flow on the plate),
                     plate_oblique.py (the near plate from an angled camera; the
                     plate gate every trace passes), frames, rotation/probing,
                     segmentation, clip cutting, uploads
client/              everything served to the browser — no Python
  templates/         index.html (Jinja)
  static/            styles.css, app.js, service-worker.js, manifest.webmanifest,
                     img/ (mark, filming guide), icons/ (PWA icons)
prompts/             model prompts, shared by the server and both CLIs
tools/               make_icons.py (PWA icons), make_demo_sample.py (the cached sample)
tests/               pytest; every Token Factory call is mocked
demo_sample/         the cached sample the public demo serves for free (contents gitignored,
                     baked into the image)
score_clip.py        CLI: one clip
score_session.py     CLI: a whole session, split into exercises
Dockerfile           the app as one container (ffmpeg and fonts included); the public demo
                     with its spend caps on
deploy/              docker-compose.yml + Caddyfile: HTTPS on a VM in one command
```

### Nebius Token Factory models (two-stage)
- Base URL: `https://api.tokenfactory.nebius.com/v1`
- API key: `NEBIUS_API_KEY` (required)
- Stage 1 (vision/perception): `openbmb/MiniCPM-V-4_5` (configurable via `NEBIUS_VISION_MODEL`) — receives sampled frames and outputs factual observations only (no scores).
- Stage 2 (judge/reasoning): `nvidia/Nemotron-3-Ultra-550b-a55b` (configurable via `NEBIUS_JUDGE_MODEL`) — receives the rubric + per-lift prompt and the Stage‑1 observations, and returns the validated JSON score.
- Judge fallback: `nvidia/nemotron-3-super-120b-a12b` (configurable via `NEBIUS_JUDGE_FALLBACK_MODEL`). If the primary judge errors, times out, or never returns valid JSON — even after the one repair retry — the same prompt is sent to this model. Set the variable to an empty value to turn the fallback off.
- The Docker image — the public demo — inverts this: its judge is **Nemotron 3 Super with no fallback**, so a failed judge call fails rather than escalating to Ultra. The verdict carries `judged_by`, the model that actually answered, and the card shows it, so neither the footer nor a result ever claims a model that was not used.
- Cosmos models are not currently served on Token Factory in this account. If needed later, Cosmos on AI Cloud is a future option outside Token Factory.

The NVIDIA Nemotron models on this account reject image input, so the perception stage runs on
MiniCPM‑V. All reasoning and scoring — the rubric, the band, the drift call, the cues — is the
NVIDIA model's, and both stages are served by Token Factory.

Notes:
- Models listing: use `GET /v1/models` to verify availability in your workspace.
- Code paths are mocked in tests; run locally with your key to verify end‑to‑end.

You can confirm availability in your workspace:

```bash
curl -H "Authorization: Bearer $NEBIUS_API_KEY" https://api.tokenfactory.nebius.com/v1/models | jq .
```

Environment variables:
- `NEBIUS_API_KEY` (required)
- `NEBIUS_BASE_URL` (default `https://api.tokenfactory.nebius.com/v1`)
- `NEBIUS_VISION_MODEL` (default `openbmb/MiniCPM-V-4_5`)
- `NEBIUS_VISION_MAX_IMAGES` (default `10`)
- `NEBIUS_JUDGE_MODEL` (default `nvidia/Nemotron-3-Ultra-550b-a55b`)
- `NEBIUS_JUDGE_FALLBACK_MODEL` (default `nvidia/nemotron-3-super-120b-a12b`; empty disables)
- `NEBIUS_JUDGE_MAX_TOKENS` (default `2048`)
- `MAX_UPLOAD_MB` (default `200`; `0` disables the upload size check)
- `JOB_TTL_SECONDS` (default `1800`) — how long a finished job and its replay are kept
- `JOB_TIMEOUT_SECONDS` (default `600`; `0` disables) — wall clock one scoring run may take,
  model calls and replay included; past it the run is cancelled and no further call is made
- Public-demo caps, off in a checkout and **on in the Docker image** — see
  [Running the public demo](#running-the-public-demo-without-a-surprise-bill):
  `DEMO_ACCESS_CODE`, `DEMO_DAILY_LIMIT`, `DEMO_IP_HOURLY_LIMIT`, `TRUSTED_PROXIES`,
  `MAX_CONCURRENT_JOBS`, `MAX_CLIP_SECONDS`, `MAX_FRAMES`, `DEMO_SAMPLE_DIR`,
  `SAMPLE_RESULT_URL`, `SAMPLE_CLIP_URL`
- `SAMPLE_CLIP_URL` — Google Drive "anyone-with-link" URL for the sample video.
  The dyno downloads the clip once (handling Drive's virus-scan interstitial),
  caches it under `SAMPLE_CACHE_DIR` (default `/tmp/lift-judge-samples`), and
  re-renders the annotated replay from the bar-path points stored in the
  committed `demo_sample/sample.json`. No video is ever committed to the repo,
  and no model call is made at serve time. `SAMPLE_RESULT_URL` — alternative to
  the committed `sample.json`, for the case where the whole bundle is hosted
  externally (JSON only, replay still rendered from `SAMPLE_CLIP_URL`).

  Build the bundle once locally and commit the JSON + evidence stills:
    ```
    python tools/make_demo_sample.py \
      "https://drive.google.com/uc?export=download&id=<file-id>" \
      --lift clean --out demo_sample/
    git add demo_sample/sample.json demo_sample/*.jpg   # NOT the mp4
    git commit -m "Build demo sample"
    ```
  Then on Heroku:
    ```
    heroku config:set \
      SAMPLE_CLIP_URL="https://drive.google.com/uc?export=download&id=<file-id>" \
      DEMO_SAMPLE_DIR=demo_sample \
      -a sideline-lift-judge-99d464f34df0
    git push heroku main
    ```
  Verify: `curl https://<app>/api/config` reports `"sample_available": true`;
  `curl -I https://<app>/api/sample` returns 200 and the daily-call counter
  (visible via `/api/config`) stays unchanged after fetching the sample.

Both stages request `chat_template_kwargs.enable_thinking = false`. The judge models are
reasoning models: left to think, they spend the whole completion budget on reasoning, get cut
off at `finish_reason: length`, and never emit the JSON object. The judge budget keeps headroom
in case a provider ignores that hint.

### Prompts
The prompts are fixed, stored under `prompts/`:
- `prompts/system.txt` (physics/rubric; Wingate principles)
- `prompts/user_snatch.txt`, `prompts/user_clean.txt`, `prompts/user_jerk.txt`
- `prompts/vision_observation.txt` (Stage‑1 factual observation instructions)
- `prompts/vision_triage.txt` (Stage‑0b per-frame bar position, used by `score_session.py`)

### Setup
1) Python 3.11+ (verified on 3.13 from a clean virtualenv)
2) Create `.env` (see `.env.example`) with at least:
   - `NEBIUS_API_KEY=...`
   - Optionally override `NEBIUS_BASE_URL`, `NEBIUS_VISION_MODEL`, `NEBIUS_JUDGE_MODEL`
3) Install deps:
   ```bash
   pip install -r requirements.txt
   ```
4) Install **ffmpeg** (provides `ffmpeg` and `ffprobe`), which is a system package, not a pip one:
   ```bash
   sudo apt-get install ffmpeg    # Debian/Ubuntu
   brew install ffmpeg            # macOS
   ```
   `ffprobe` reads the rotation metadata that portrait phone video carries. Without it
   rotation detection silently returns 0°, and portrait clips are judged sideways —
   so install it before testing with phone footage.

### Run the web app
From the repo root, with `NEBIUS_API_KEY` set (in `.env` or the environment):

```bash
uvicorn server.main:app --reload --port 8000
```

Then open **http://localhost:8000**. To reach it from your phone on the same wifi, bind to all
interfaces and browse to your machine's LAN address:

```bash
uvicorn server.main:app --host 0.0.0.0 --port 8000
```

> Over plain HTTP, `localhost` is treated as secure but a LAN address is not, so the camera and
> PWA install will not work there — everything else will. For the real phone test you need the
> HTTPS deploy.

The flow is four screens:

1. **Choose the lift** — snatch, clean, jerk, or *Find them* to scan a longer clip for every attempt.
2. **Filming guide** — shown *before* the camera opens, not after a bad result.
3. **Film or upload, then trim** — start/end sliders with a preview loop; trim times go to the
   server, so nothing is re-encoded on the phone.
4. **Score** — staged progress while it works, then the verdict.

The result screen shows the score and band in large type, the max drift, a line saying which
model did what, a confidence meter with any warnings next to it, 1–2 cue cards, collapsible
notes, and the annotated replay with download and share buttons.

The drift line is **written from the tracker's measurement** whenever the bar was followed —
*"Noticeable drift forward: about 9 cm (0.21 plate widths) off the start line, furthest around
the knee, in the transition."* — so the words on the card and the number under them are the same
fact; the judge's text lends only the direction, which the tracker cannot know. When the bar was
not tracked the line is the judge's own description, labelled as such. The judge's words on the
drift are kept under *Notes* either way, so nothing the model said is hidden.

Under it, in small type: *Bar tracked locally (OpenCV, no model), plate seen in 93% of frames ·
Frames read by MiniCPM-V 4.5 · Judged by NVIDIA Nemotron 3 Super* — with *(fallback judge)*
appended when the primary failed and the second model answered. The names come from the
server's record of which model actually replied (`models` and `judged_by` in the result), not
from a constant.

Under the hood each run: finds where the plate rises inside the trimmed window and follows it
(local OpenCV), samples 8–16 frames (default 10) from that lift, applies phone rotation metadata,
downsizes to ~768 px, asks the vision model what the bar did, hands the Nemotron judge those
observations **and the tracker's measurement** to score against the rubric, caps the confidence
against clip quality and against the measurement, and renders the replay with the same tracked
path drawn on it.

**Camera capture and PWA install both require HTTPS** (`localhost` is exempt). Behind a reverse
proxy, terminate TLS there and allow a request body at least as large as `MAX_UPLOAD_MB`.

#### Installing it on a phone
The app ships a web app manifest, a maskable icon and a service worker that caches the shell, so
it installs to the home screen and opens full-screen. Android Chrome offers an install prompt;
iOS has none, so use Share → *Add to Home Screen* (the app shows a hint). Scoring always needs the
network — nothing under `/api` is ever cached.

#### Asynchronous scoring
The upload returns a job id and the page polls it, so a slow or dropped mobile connection cannot
lose a result:

| Endpoint | Purpose |
|---|---|
| `POST /api/jobs` | `video`, `lift`, `mode`, `trim_start`, `trim_end`, `frames` → `202 {id, poll_url}` |
| `GET /api/jobs/{id}` | status, stage, message, progress, and the lifts once done |
| `GET /api/jobs/{id}/replay/{n}.mp4` | the annotated replay for lift *n* |
| `POST /api/score` | synchronous one-shot scoring, for scripts |

When the demo is gated, both `POST`s take the code as the `X-Access-Code` header (a form field
is inside the body, which the check is there not to read). Jobs and their replays are deleted
after `JOB_TTL_SECONDS` (default 30 minutes), the uploaded clip is deleted as soon as scoring
finishes, and a run is cancelled after `JOB_TIMEOUT_SECONDS`.

### Filming guide
The score is only as good as the angle. The app shows this before the camera opens, and it is
worth repeating here:

- Film **square to the side** — not in front, not at an angle. Drift is measured sideways.
- **3–5 m away**, camera at **hip height**.
- **Portrait is fine.** Framing is what matters, not aspect ratio.
- Keep the **whole bar and both plates in frame** for the entire lift. The bar leaving the frame
  is what caps confidence, not the orientation.
- **One lift per clip.** Trim it on the trim screen; a clean & jerk is scored as two lifts, and
  a set of reps is scored rep by rep (each rise that starts back on the floor is a lift).
- **Filmed from an angle anyway?** After the trim, tap the plate nearest the camera on the first
  frame. The tracker can follow that plate as the ellipse it appears to be, measures the angle
  from its shape and caps the confidence for it; the tap only tells it which plate you mean. If no
  plate can be followed, the card says *Couldn't track the bar — film from the side with the near
  plate visible* and no path is drawn.

### CLI (no UI)
```bash
python score_clip.py /path/to/clip.mp4 --lift snatch
python score_clip.py clip.mp4 --lift clean --trim-start 2.5 --trim-end 7 --debug
python score_clip.py clip.mp4 --lift snatch --replay-dir out/
```
Options:
- `--lift snatch|clean|jerk`, or `--auto` to let the vision model find the lifts
- `--frames 10` (clamped to 8–16; at most 10 images per vision call; extra frames are chunked)
- `--trim-start` / `--trim-end` in seconds
- `--no-split` to skip the triage call that separates a clean & jerk
- `--replay-dir DIR` to render the annotated MP4
- `--prompts-dir prompts/` (`--prompts_dir` still works)
- `--debug` (Stage‑1 observations to stderr), `--progress` (stages to stderr)

Outputs the validated JSON to stdout. It runs the **same pipeline as the web app**, so a result
here and a result on the phone cannot drift apart: triage, observations, the judge with its
repair retry and fallback model, and the clip-quality confidence rule. A clip holding a clean &
jerk prints both lifts.

### Session CLI — split a clip into exercises, then rank them
`score_clip.py` judges one lift in one clip. `score_session.py` takes a clip holding several
exercises, cuts it into one window per exercise, works out which lift each one is, scores
each separately and prints them best-first.

```bash
python score_session.py samples/session.mp4 --out-dir out/
```

Four stages, two of them free:

| Stage | What it does | Cost |
|---|---|---|
| 0a motion scan | OpenCV frame-differencing finds bursts of activity separated by idle gaps → candidate windows | local, no API call |
| 0b triage | The vision model reports **where the bar is in each frame** (floor / knees / shoulders / overhead); the lift is derived from that path in code | 1 vision call per window |
| 1 observations | The usual Stage‑1 factual bar-path description, per exercise | 1 vision call per exercise |
| 2 judge | The usual Nemotron JSON verdict, per exercise | 1 judge call per exercise |

Triage asks for bar positions rather than a lift name on purpose: asked to name the lift,
MiniCPM‑V answered "clean and jerk" with confidence 1.0 for every window of a clean & jerk
clip, including the one where the athlete was only dumping the bar. Asked where the bar is in
each numbered frame, it answers from the pixels, and the classification follows from the path:

- floor → overhead, no rack on the way up = **snatch**
- floor → shoulders = **clean**
- shoulders → overhead = **jerk**
- floor → shoulders → overhead = **clean & jerk**, split at the quietest moment between the
  two and scored as two exercises (per spec §3)
- bar never rises, or only comes down (a drop, a re-rack, an idle platform) = **skipped**, and
  listed under "Skipped windows" with what the model saw

A level has to show up in at least two frames to count, so one misread frame cannot turn a
clean into a jerk.

Outputs:
- `out/report.json` — every window with timestamps, lift, full judgment, and the windows that
  were skipped and why
- `out/exercise_NN_<lift>.mp4` — each exercise cut out with ffmpeg stream-copy
- `out/exercise_NN_frames/` — the frames that were actually sent to the vision model
- a ranked table on stdout (best score first; unscored exercises sort last)

Without `--out-dir` or `--json`, the report goes to stdout (pipeable into `jq`) and the table
to stderr.

Options:
- `--lift snatch` — force every window to one lift and skip triage entirely (cheaper, and
  right when you filmed a set of the same lift)
- `--split-only` — print the windows and exit, no API calls at all; use it to tune splitting
- `--sensitivity 0.25` (0–1, lower splits more eagerly), `--min-duration 0.6` (minimum
  high-motion core, not window length), `--max-duration 20`, `--merge-gap 1.0`, `--pad 0.8`,
  `--sample-fps 8`
- `--max-segments N` — keep only the N busiest windows
- `--concurrency 2` — exercises scored in parallel
- `--no-clips`, `--no-frames` — skip those outputs
- `--frames 10` (clamped to 8–16), `--debug` (triage replies + Stage‑1 observations on stderr)

Tuning tip: start with `--split-only`. If an exercise is missing, lower `--min-duration` or
`--sensitivity`; if one exercise arrives as two windows, raise `--merge-gap`. A clip with no
quiet/busy contrast at all falls back to a single window, so a plain one-lift clip still works.

One exercise failing to score never sinks the run — the error lands in that exercise's `error`
field and the rest still rank.

### Confidence, and when not to trust a score
The judge only ever reads a text description of the clip, so it cannot tell that the camera was
off to one side or that the window held two lifts. Left alone it once returned **0.85 on a clip
that mixed a clean and a jerk**. So the server caps confidence from facts it can check, and every
cap comes with a warning shown next to the score:

| Problem | Confidence capped at |
|---|---|
| The window holds more than one lift and could not be split | 0.35 |
| The bar was not visible in part of the clip | 0.40 |
| Camera not square to the side | 0.50 |
| The verdict contradicts the tracked path (see below) | 0.50 |
| Occlusion, darkness or motion blur | 0.55 |
| Phases not clearly separable | 0.60 |
| Clip too short to hold a lift | 0.60 |
| The bar left the frame during the lift | 0.60 |
| The plate is small on screen (hard to track to a few pixels) | 0.65 |
| The tracker could not follow the bar, so the drift rests on the frames alone | 0.65 |
| Only a few frames readable | 0.70 |
| The tracker saw the plate in under 75% of the sampled frames | 0.70 |

Caps are ceilings, never floors — a judge that is already unsure stays unsure. The worst
applicable cap wins. A clean & jerk that *was* split is noted, not penalised.

**The re-check.** After the judge replies, the verdict is held against the measurement: a path
scored 7+ while the plate was measured half a diameter or more off its starting line, or scored
4 or under while it barely moved sideways, earns a warning that names the number and a cap at
0.50. It only runs when the tracker saw at least 90% of the lift and the view is not flagged as
off-axis (parallax inflates the measured drift from an oblique camera).

### Annotated replay and the bar-path trace
Each scored lift gets an MP4 with the verdict burnt in — lift, score, band, confidence and the
first cue — so the file still carries the score after it goes through a share sheet.

When the barbell can be followed confidently, the replay also carries the **bar path** itself, so
the cue has something to point at:

| Mark | Meaning |
|---|---|
| Green line | the path the bar actually took |
| Dashed white line | where the bar started — what "keep the bar close" is measured against |
| Orange ring and ticks | the point where the bar strayed furthest from that line |

Tracking is local OpenCV (`server/utils/bar_track.py`) — no model call and no extra token cost. It
finds circles that move, follows several candidates at once through a velocity prediction that
survives frames where Hough misses the plate, and keeps the one that behaves like a barbell: a
plate travels a long way up the frame, while a head or a plate parked on a stand does not. Plate
colour enters the walk's cost so the trail prefers the plate it started on over the far plate.

Which trail stands for the bar is decided by agreement, not by the biggest rise: several seeds on
the plate walk onto the same circles frame after frame, while a trail that hopped from a logo on
the wall banner to the plate overhead agrees with nobody — and that hop is exactly what used to
win on "biggest rise". Among the trails with company, the biggest rise over the most of the window
wins, which from an off-axis camera is the near plate: the one the athlete's legs never hide, so
the trace starts on the floor at lift-off instead of appearing above the knee.

It gives up rather than guess. If the bar is found in under ~55% of frames, barely rises, or never
moves, no trace is drawn and the replay is the verdict-only version. `bar_path_tracked` in the
result says which you got, and `bar_path.reason` why not.

**From an angled camera** (`server/utils/plate_oblique.py`) the near bumper is an ellipse —
taller than wide, axis ratio ~0.6 from 45° off — that no circle detector finds, so the followers
above used to settle on the smaller far plate, or on foliage in a window behind the lifter, and
"tracked" could not tell. A third follower runs Hough on the frame stretched sideways by 1.0,
1.3, 1.6 and 2.0 (an ellipse of ratio *q* is a circle at a stretch of 1/*q*), keeps only candidates
that are dark inside with a rim all the way round, walks forward from the plate on the floor and
backward from everything found high in the frame — a backward walk counts only when it lands back
on the floor plate, which a ring hanging above the lifter or the far plate never does — and keeps
the trail that rose furthest in pixels (the near plate rises more than the far one). The axis
ratio it reports is an ellipse fit on the rim, with 1/stretch as the fallback, and feeds the
off-axis confidence cap. It costs one full-frame scan every third frame at 12 fps and a 480 px
long side, a patch around the prediction on the others, everything cached per frame: ~0.3 s of
analysis on one thread for a 4 s 1080p window once decoded (`tools/bench_oblique.py`).

**Every other trace passes a plate gate** (`looks_like_a_plate`) before it is drawn: dark inside
with a rim, or a near-complete rim whatever the colour (white and yellow competition bumpers).
The thresholds come from 522 samples along the hand-labelled plate tracks of 44 clips: real
plates read a median grey of 100 inside (p90 128 — coloured bumpers, hub lettering, gym light),
so the gate sits at 130, not at the 90 a single black bumper suggests. The order is: the side-on
follower's trace when it passes the gate — unless the oblique follower measured the view as
oblique *and* found a different, clearly bigger plate, the near one it could not see — then the
oblique follower's, then the lift-window trail and the legacy tracker, each gated. When nothing
passes, the card says *Couldn't track the bar — film from the side with the near plate visible*.

**The tap.** After the trim the app shows the first frame of the window and takes one tap on the
plate nearest the camera, sent as the form field `plate_tap="t,x,y"`: t in seconds of the file
that went up (0 when the browser re-encoded the trim, the trim start otherwise), x and y as
fractions of the frame as the phone shows it, turned back onto the decoded frame when OpenCV hands
back the unrotated one. The followers seed the frame it was made on. It is a preference, not an
answer: it decides between two plates that both rose and cannot make something that did not rise
the bar. A missing or malformed field is the automatic search, never a refused upload. When the
plate followed is the tapped one the result says so (`bar_path.tapped`, and *Bar path: following
the plate you tapped* on the card). Skip it and the tracker picks the plate itself.

### What the judge is told, and what the number means
The judge never sees pixels. Before any model call, the tracked path is turned into a short
evidence block (`server/services/measurement.py`) and sent to Nemotron alongside the vision
observations:

- how far the bar strayed from the line it started over, in **plate diameters** — the one length
  whose real size is known in every clip — and roughly in centimetres for a 450 mm bumper;
- **where in the rise** that happened (first pull, around the knee, second pull, turnover; dip,
  drive, lockout for a jerk);
- the sideways range, how far the bar rose, and in what share of the sampled frames the plate
  was seen;
- a rule of thumb for a side-on clip (under 0.15 diameters is close, above 0.35 is a large
  drift) and a warning that an oblique camera inflates the figure.

The judge is told what each source is good for: the measurement for *how far* and *where*, the
observations for *direction* (forward or back), bar–body contact and phase quality, and to say in
its notes when it prefers one over the other. The same numbers come back in the result as
`bar_path`, and the card's drift line (`drift` in the result) is composed from them on the server
— size band, centimetres, plate widths, where in the rise — with the judge consulted only for
forward or back. So the judge can no longer put "minimal drift" above a ring drawn half a plate
off the line; its words go under *Notes*, and the re-check still flags the disagreement. When
the bar could not be followed, the judge is told that too and is asked to keep its confidence
down, the server caps it anyway, and the card's drift line is the judge's read, labelled as such.

Read the number for what it is. It is measured from one plate's apparent position, so parallax,
an off-axis camera and the athlete moving across the frame all leak into it; the centimetre figure
assumes a competition bumper. It is the honest size of the drift *in this clip as filmed*, not a
lab measurement — which is why it is labelled a tracker estimate and why an off-axis clip caps the
confidence regardless.

### Finding the lifts in a clip
The same tracker decides *where the lifts are*. A lift is the plate going up a long way in a few
seconds; walking in, chalking up, re-racking and dropping the bar are not, however much they move
the picture — which is why the earlier motion-energy segmentation kept picking the bar drop and
the setup. `detect_lift_windows` analyses the clip in overlapping 8-second chunks, follows every
plausible plate in each, and turns each credible rise into a padded window. "Credible" is learned
from 39 real clips: the plate is detected in most frames of a real lift, it barely wanders
sideways relative to how far it rises, and its circle is the same size as the clip's plate — a
rising circle half or twice that size is a torso, a head or a logo on a wall banner. Rises less
than ~1.5 s apart merge as the phases of one lift (pull, catch, recovery; a clean and the jerk
after it) — unless the later rise starts back on the floor, within 35% of the earlier lift's
height above it: that is the next rep of a set, and it gets a window of its own however soon it
comes (the jerk starts from the dip, 80–90% of the way up, and stays with its clean). No window is
allowed to be longer than one lift. The motion-energy segmentation remains only as a fallback
when nothing could be tracked.

**Measured on 39 real clips** (63 hand-labelled lifts, `tests/data/lift_windows_truth.json`,
re-measured 2026-10-28): recall 89% (7 lifts missed), precision 86% (7 junk windows of 50). The
gain over the 85% / 78% of 2026-10-21 came from reading frames on their own presentation time:
9 of the 39 clips are variable frame rate, and frame number / fps had put their windows up to
two seconds from the lift (see `server/utils/frames.py`).
The misses are mostly jerks whose drive is so fast that the plate is motion-blurred for the whole
0.3 s at 12 fps; the extra windows are an athlete walking in or the bar being dropped or
re-racked, which the triage stage rejects before anything is scored. Two windows merge when
their padding touches, and a joined window never outgrows one lift: when it would, the later rise
starts a window of its own at the boundary instead of being absorbed and truncated away (which
used to leave a window whose own peak lay outside it, with the other attempt's trail inside).
`tests/test_lift_windows_real.py` re-measures this when the clips are present and fails below
80/75.

This runs in *single* mode too: inside whatever you trimmed, the sampled frames come from the
rise, not the setup, and a clean & jerk comes out as its two rises.

When triage calls a window a clean & jerk, the window's own trail decides where to cut: two
rises split between them; one rise falls back to the quiet moment in the motion profile; and no
split may produce a "lift" shorter than 1.2 s, which is what a slice of the clean looks like. If
the window cannot be split at all, the first lift is scored and the confidence is capped with a
warning, because the frames hold both.

A known limit: the vision model reads a deep snatch catch — bar overhead, athlete in a squat —
as "shoulders", so some snatches are called clean & jerk. The bar's travel cannot settle it (a
clean & jerk done in one motion is ~1.5 s floor to overhead, like a snatch), so the model's call
stands and the mixed-lift cap applies. Choosing the lift yourself on the first screen avoids
this entirely; *Find them* is for session clips.

**Replay encoding** is deliberately conservative so iOS Safari plays it inline: H.264, `yuv420p`,
even dimensions, AAC or no audio, `+faststart`, long side capped at 720 px. If no usable font is
found, it falls back to the same clip without the overlay rather than to no replay at all.
Requires ffmpeg.

### Running the public demo without a surprise bill
Token Factory bills per call, and a demo URL in a submission is open to anyone who finds it.
The protection is in layers, so that even the worst case is a few dollars.

**Cap the money first, at the account.** Nothing in the app can beat this layer.
- Keep only prepaid credits on the Nebius account — no card — so when the credits run out,
  calls fail instead of billing you.
- Make a separate API key just for the demo, so it can be revoked in seconds without
  touching the key you record the video with.
- Check Token Factory usage daily until judging ends, then shut the demo down or revoke the key.

**Then limit usage in the app — which the deploys do by default.** In a local checkout every
cap below is off until its variable is set, so local runs, the CLIs and the tests are untouched.
The Docker image and `deploy/heroku-web.sh` are the public demo, so there the caps are set with
the values in the table; a container's own environment (`--env-file`, `-e`, a Heroku config var)
overrides any of them. A refused request gets one plain sentence that says what to do next, a
`Retry-After` header and a machine-readable `reason` the page acts on — and the sentence never
names a variable, a model or a call count, because a visitor can act on none of those.

**Three ways in, one variable.** `ACCESS_MODE` is `open` (no code anywhere), `link` (the code
rides in the `?code=…` link the page is opened from, is stored on the device and sent silently,
never typed; a request without it gets a friendly 401 and the sample) or `code` (typed on the
first screen, validated on *Next* before anyone films — the default when unset). Switching is
`heroku config:set ACCESS_MODE=…`, no code change. In `open` mode the app **refuses to start**
when a Token Factory key is set and any cap in the table is unset or zero, listing which, so a
deploy that forgot one never runs uncapped. The public demo for the submission runs `open`.

| Layer | Variable | In the deploys | What the visitor sees |
|---|---|---|---|
| Access mode | `ACCESS_MODE` | unset (`code`) until the rollout flips it to `open` | nothing in `open`; the footer says which mode is on |
| Access code | `DEMO_ACCESS_CODE` | **you set it** for `link` and `code` | `link`: nothing, or "This demo opens from the link on the submission page" with the sample button. `code`: a field on the first screen. Either way checked from the `X-Access-Code` header **before the upload is read** |
| Kill switch | `DEMO_DISABLED` | unset | `true`: "Live scoring is paused right now", the start screen offers the sample only, every scoring request is 503 — no redeploy |
| Daily budget | `DEMO_DAILY_LIMIT` | `150` model calls | "Today's free demo limit is used up. See the sample result, or come back tomorrow." (HTTP 429) |
| Day boundary | `DEMO_DAY_TIMEZONE` | `Asia/Jerusalem` | nothing — the budget resets at local midnight (UTC when unset) |
| Newcomer share | `DEMO_DAILY_RESERVE_PERCENT` | `20` | an address that already scored today stops 20% short: "…what is left is kept for first-time visitors" — a judge arriving in the evening still gets in |
| Per-IP rate | `DEMO_IP_HOURLY_LIMIT` | `12` model calls | "You've hit the hourly limit. Try again in N minutes or see the sample." |
| Whose address | `TRUSTED_PROXIES` | `*` on Heroku, `private` in compose | nothing — decides whether `X-Forwarded-For` is believed, see below |
| In flight at once | `MAX_CONCURRENT_JOBS` | `1` | "Another lift is being scored. You're next; this usually takes under a minute." The page retries by itself when the slot frees |
| Run deadline | `JOB_TIMEOUT_SECONDS` | `120` (600 anywhere else) | "Scoring took too long and was stopped. Trim the clip to one lift and try again." The slot is freed |
| Clip length | `MAX_CLIP_SECONDS` | `15` | the end handle starts at 15 s and the score button waits until the window fits; the server says "Clips need to be under 15 seconds" to anything longer |
| Upload size | `MAX_UPLOAD_MB` | `60` | "Clips need to be under 60 MB" (HTTP 413); the page warns from 30 MB |
| Lifts per clip | `MAX_LIFTS_PER_CLIP` | `2` | the first two lift windows are scored, the rest listed under *Windows skipped* |
| Frames per lift | `MAX_FRAMES` | `10` | nothing — one vision call per lift instead of two |
| Cheaper judge | `NEBIUS_JUDGE_MODEL` | `nvidia/nemotron-3-super-120b-a12b` | the footer and every card name the judge that actually answered |
| No fallback | `NEBIUS_JUDGE_FALLBACK_MODEL` | empty | a failed judge call is an error, never a quiet escalation to Ultra |
| Free fallback | `DEMO_SAMPLE_DIR` | `demo_sample` (shipped with the deploy) | **See a sample result**: the first button on the first screen, and next to every refusal. Never counted, never gated, still there when the demo is paused |

**The budgets count model calls, not uploads.** A single-lift clip costs three (triage,
observation, judge); a repair retry or the fallback judge adds one each, and a *Find them* clip
adds one triage per candidate window — so an upload is anywhere from three calls to a dozen,
and counting uploads left the bill open-ended. A run is admitted when there is room for one
whole lift — checked from the request headers, before the upload is read, so a clip that cannot
be scored is never received — and once the local tracker has found the clip's lift windows the
run must fit the budget whole (one lift's calls per window) or it stops **before the first
model call**: "This clip holds 2 lifts, more than today's free demo has room for. Trim it to one
lift…". It is then charged call by call (retries included, since each attempt is billed). If the
budget still runs out part-way — a window that turned out to hold a clean *and* a jerk — the run
stops at that call; the lifts already scored are kept and the one that was not says why. The
cap is on the *scored window*, not the upload: a long gallery clip trimmed to one lift is fine.
Several access codes can be comma-separated, so one can be revoked without the rest. Counters
live in the process, like the job store, so a restart resets the day's count — which errs on
the side of letting a judge in. The sample, the polls and the replays are never counted.

**What the log says.** One line per admitted run — `job status=done ip=<keyed hash>
first_today=yes calls=3 remaining_today=147 running=0 duration=41.2s` — with a hash of the
address keyed per process, never the address, and never the clip's name. `GET /api/config`
reports the mode, the caps and the calls left today (no secrets), which is the quickest check
after a deploy. In `open` mode `/docs` and `/openapi.json` are off.

**Whose address the per-IP limit sees.** `X-Forwarded-For` is only believed when the
connection comes from a proxy named in `TRUSTED_PROXIES` — addresses, CIDR ranges, `private`
(loopback, RFC 1918, link-local: a container network or a proxy on the same host) or `*`
(whatever reaches the port; only right when nothing but your proxy can, as on Cloud Run or
Fly). Behind a trusted proxy the visitor is the rightmost entry the proxy appended; anything a
client put in the header itself is ignored. Unset, the header is ignored entirely and the
socket's peer is the visitor — a direct caller can no longer pick a fresh hourly allowance by
forging the header, but behind a proxy *everyone* would then share the proxy's one allowance,
which the startup log warns about. The **first** entry is never believed on its own: ten
requests with ten forged first entries are limited exactly like ten with none (there is a test).
`deploy/docker-compose.yml` sets `private` for Caddy and Heroku gets `*`; the image never tells
uvicorn to trust the header from everyone.

**Have a fallback that costs nothing.** Score one clip once, ahead of time, and the app serves
that result — replay included — from disk, labelled *Sample result (pre-recorded)*. The sample is
**part of the deploy**: build it, and either commit `demo_sample/` (the one video the repo
allows, because the demo serves it to everyone anyway — keep it to one lift, under ~10 MB) so a
`git push` deploy carries it, or build the image after it so the image carries it at
`/app/demo_sample`. Then check that `/api/sample` and its replay answer 200 on the live URL.

```bash
python tools/make_demo_sample.py samples/snatch1.mp4 --lift snatch --out demo_sample/
git add demo_sample/ && git commit -m "Add the cached demo sample"     # or: docker build -t lift-judge .
```

The button is the first thing on the first screen, and appears again next to every refusal, so
a judge who arrives after the budget is spent — or while the demo is paused — still sees the
whole result screen in two taps and under 30 seconds. The sample needs no access code and is
not counted, and its card names the judge that scored it when it was made (Ultra, if you built it
from a checkout with the defaults) — which may differ from the live judge, and that is the truth
of it. Pick a clip you are allowed to show in public. The server logs a warning at startup when
caps are set and there is no sample, and `deploy/heroku.sh` refuses to deploy `open` without one.

**Keep Ultra for your own recorded run.** The image runs Nemotron 3 Super with no fallback;
the code defaults — what a local checkout and the video you record use — stay on Ultra with a
Super fallback. The footer and every verdict card name the judge that actually answered, so the
demo never claims a model it is not using.

**Read the first lines of the log.** At startup the server prints the access mode, which judge,
fallback and caps are in force, where the day turns, which proxies are trusted and whether a
sample is present, and warns when caps are set in `code` mode without a code, without a sample,
or with a per-IP limit and no trusted proxy. In `open` mode a missing cap is not a warning but
a refusal to start.

**The rollout, and the way back.** Ship the sample and the caps with `ACCESS_MODE` unset (code
mode); verify the live URL — `/api/sample` 200, `/api/config` showing the caps, the sample in two
taps; then `heroku config:set ACCESS_MODE=open`. Watch the log lines and Token Factory usage for
a day. If spend or abuse looks wrong, `ACCESS_MODE=link` and put the `?code=` link in the
submission; if needed, `DEMO_DISABLED=true`. After judging: revoke the demo key and set
`DEMO_DISABLED=true`. None of these steps is a deploy.

### Upload limits and errors
- Uploads are streamed to disk and rejected over `MAX_UPLOAD_MB` (HTTP 413) with "Clips need to
  be under 60 MB. Trim this one to the lift in your gallery app and try again."; the partial
  file is removed. A window over `MAX_CLIP_SECONDS` gets "Clips need to be under 15 seconds".
- Any `video/*` type is accepted, plus generic types (e.g. `application/octet-stream`) that
  carry a video extension — phones are inconsistent about what they send.
- Any failure reaches the phone as one plain sentence that says what to do next ("The model
  service is busy right now. Wait a minute and try the clip again."), never the provider's
  reply or a stack trace; those go to the server log, and to `error_detail` for the CLIs
  (`server/services/errors.py`).
- Token Factory calls connect within 10 s and must answer within `NEBIUS_TIMEOUT_SECONDS`
  (default 45; a healthy call takes 2–3 s). A 429 or 5xx gets exactly one more try after a short
  pause; the judge then falls over to its second model, if one is configured. Dropping the
  no-thinking hint and asking again happens only when the provider rejects the parameter
  (400/422), not on an outage.
- A whole run — tracking, every model call, the replay render — must finish within
  `JOB_TIMEOUT_SECONDS` (default 600, the deploys set 120). Past that it is cancelled, no further
  model call is made, the slot is freed, and the phone reads "Scoring took too long and was
  stopped." The synchronous endpoint answers 504.
- The demo's gate — the kill switch, the access code in `link` and `code` modes, a free slot and
  room in the budgets for one lift — runs in ASGI middleware on the `X-Access-Code` and
  `X-Forwarded-For` headers, before FastAPI reads the multipart body. A clip that cannot be
  scored is refused from its headers alone and never spooled to disk.
- A job that a limit stops carries the limit's `reason` in its status, so the page offers the
  right way on: the sample for the day, a countdown for a busy server, the same clip again for
  anything else.
- The uploaded clip is deleted as soon as scoring finishes, pass or fail.

### Local samples (do not commit videos)
- Put your test clips under `samples/` (this path is gitignored).
- Example:
  ```bash
  mkdir -p samples
  cp /path/to/your/side_view_snatch.mp4 samples/snatch1.mp4
  python score_clip.py samples/snatch1.mp4 --lift snatch
  ```
- Videos must not be committed to the public repo.

### Tests
Run unit tests:
```bash
pytest -q
```
Coverage in this MVP:
 - Frame sampling: creates a dummy video and verifies even sampling
- JSON validation: success cases, fenced JSON, cues count logic, and band=unknown special-case
- Nebius API calls: two-stage flow mocked to validate both the observation and judge steps (no live inference)
- Splitting: builds synthetic session videos and checks that each busy window is found, that
  close windows merge, that fidget-length ones are dropped, and that a contrast-free clip
  still yields one window
- Session pipeline: a whole `score_session.py` run with every model call mocked — triage,
  skipping, clean & jerk splitting, ranking order, per-exercise error handling and the
  judge-model fallback
- Bar-path classification: the exact per-frame readings MiniCPM‑V returned for the sample
  clip, so a regression in the derivation shows up as a failing test rather than a wrong score

### Cost note
- Frames per call: vision accepts at most 10 images per request (configurable via `NEBIUS_VISION_MAX_IMAGES`, default 10). If more frames are requested, the app chunks them across multiple vision calls and merges observations before judging.
- A session run costs one extra vision call per candidate window (triage, 8 frames) on top of the
  usual two calls per scored exercise. `--lift` skips triage; `--split-only` calls nothing.
- In the web app a single-lift clip is three calls (triage, observation, judge); a repair retry
  or the fallback judge adds one each. The public demo's budgets count exactly these.
- Token estimate: depends on provider accounting for multi-image inputs; exact tokenization is provider-specific and not documented here. Treat as unknown and verify in your Nebius Token Factory usage dashboard.
- For a URL that is open to the public, cap the spend — see
  [Running the public demo](#running-the-public-demo-without-a-surprise-bill).

### Limitations
Read these before trusting a number. Where a limitation has a real case behind it, the case is
given: these are things that happened on the 39 clips, not hypotheticals.

**What the score is**
- **Scores are indicative.** The judge reads a vision model's description of 8–16 sampled frames
  plus a drift figure tracked from one plate's apparent position. That figure is scaled by the
  plate's size, not calibrated against the scene, so parallax and an off-axis camera leak into
  it; there is no pose estimation. Treat the score as a coach's first read, not a lab result.
- **The judge's score varies by about ±1 between runs on the same clip.** The vision model's
  frame sampling and the judge's generation are both stochastic, so two scorings of the same
  lift can disagree by a point. The measured drift stays stable to a centimetre or two; it is
  the judge's rounded 1–10 verdict that moves. Shown here so a card is read as "around 6",
  not "exactly 6".
- **One scored clip is about three model calls** (vision observation + judge + repair/evidence),
  so the live demo's daily cap of 150 model calls is roughly 50 clips before the sample becomes
  the only answer. The counter is per-process and resets with the dyno; `heroku ps:restart`
  resets it on purpose. (Measured: a single untrimmed clean upload took the counter from 150
  to 147.)
- **The tracked point can sit a few centimetres off the plate's sleeve.** The plate centre is
  taken from a Hough circle plus an ellipse fit on the rim pixels, which is accurate to a few
  pixels on side-on footage and to a bit more on oblique clips. The drift arrow stays
  reliable for coaching cues; absolute centimetres on an off-axis clip are indicative.
- **Oblique camera angles overstate forward drift.** The tracker reports the plate's apparent
  horizontal position; if the camera is not square to the side, an athlete stepping toward or
  away from the lens reads as sideways travel and the confidence cap says so — but the
  centimetre number is still the one the judge saw and can be visibly wrong on an off-axis clip.
- **Framing is the quality signal.** Portrait or landscape is not what the confidence cap
  checks — the bar leaving the frame or the plate being too small on screen is. Portrait with
  the whole lift in frame scores at full confidence.
- **The centimetres assume a 450 mm competition bumper.** Training plates are smaller, so on
  them the figure overstates the drift by the same ratio. The plate-width figure is the one that
  does not depend on the assumption.
- **Direction is not measured.** The tracker gives size and height only; whether the bar went
  forward or back comes from the vision model's words. The card's "forward" can be wrong while
  its centimetres are right.
- **Angle decides everything.** A clip that is not square to the side cannot show horizontal drift
  honestly — from an oblique camera, the athlete stepping toward or away from the lens reads as
  sideways travel. The confidence cap says so; it cannot rescue the angle.

**Finding the lift (the local tracker)**
- **Measured, not assumed: 89% recall, 86% precision** on the 39 real clips (63 hand-labelled
  lifts, 2026-10-28). 7 lifts were missed — mostly jerks whose drive is so fast the plate is
  motion-blurred for the whole 0.3 s at the 12 fps the tracker samples — and 7 of 50 windows were
  junk: an athlete walking in, the bar being dropped or re-racked. Triage rejects the junk before
  anything is scored, but each rejected window still costs one vision call. The earlier figures
  of 92/84 did not reproduce and were corrected.
- **The drawn bar path** is a second pass inside each window (`server/utils/plate_follow.py`):
  the plate followed frame by frame with optical flow, not the 12 fps detections that found the
  lift. Scored against per-frame plate tracks checked by eye (`tools/score_bar_path.py`,
  re-measured 2026-10-31): 44 of 61 gym lifts get a clean trace, 95% of trace points are on the
  path, and the drift shown is off by 6 cm at the median, 16 cm at p90 and 23 cm at worst. The
  plate gate changed none of the clean counts and took the worst case from 134 cm to 23 cm: the
  two traces it removed were on nothing (52% and 6% of their points on a plate, a 140 cm drift),
  and one mediocre trace (67%) went with them. On 6 competition clips from YouTube
  (`--set competition`) 9 of 11 lifts are clean and the drift is off by 5 cm at the median.
- **A clean & jerk done fast is one rise.** Rises under ~1.5 s apart merge as the phases of one
  lift, so a jerk that follows the clean at once lands in the clean's window and the split falls
  back to the quiet moment in the motion profile. If no split leaves each half at least 1.2 s —
  what a slice of the clean looks like — only the first lift is scored and the confidence is
  capped with a warning, because the frames hold both.
- **Round things that are not the plate.** The tracker follows moving circles, and has been seen
  to lock onto a logo on a wall banner (a trail hopped from the logo to the plate overhead and won
  on "biggest rise" until agreement between trails became the rule), a plate parked on a stand
  beside the athlete, and the *far* plate on an off-axis camera — hopping from one plate to the
  other invents a sideways move that never happened. Colour and size checks now catch most of
  these, every trace must look like a plate (dark inside with a rim, or a full rim) before it is
  drawn, and from an angled camera the near plate is followed as the ellipse it is
  (`server/utils/plate_oblique.py`); a gym full of circular signage is still the hard case, and a
  bright plate with a broken rim (a hand over it, a collar) can fail the gate and leave no trace.
- **The oblique follower is tested on synthetic clips only.** No labelled oblique footage exists
  in the sample set, so its accuracy figures (every point within half a plate radius, axis ratio
  within 0.05 of the truth) are from drawn clips; on the real side-on clips it finds the same
  plate as the side-on follower and gives way to it.
- **It gives up rather than guess**: no trace when the plate is found in under ~55% of frames,
  barely rises, or never moves. Then the drift is read from the frames alone, the judge is told,
  and confidence is capped at 0.65.

**The models**
- **Triage misreads a deep snatch catch.** Bar overhead, athlete in a full squat reads to
  MiniCPM-V as "shoulders", so some snatches come back as clean & jerk. The bar's own travel
  cannot settle it (a clean & jerk done in one motion is ~1.5 s floor to overhead, like a
  snatch), so the model's call stands and the mixed-lift cap applies. Choosing the lift yourself
  on the first screen avoids this entirely; *Find them* is for session clips.
- **Asked to name the lift, the vision model names what it expects.** An earlier triage prompt
  that asked "which lift is this?" got "clean and jerk" at confidence 1.0 for *every* window of a
  clean & jerk clip, including the one where the athlete was only dumping the bar. The prompt now
  asks where the bar is in each numbered frame and derives the lift in code — and at 512 px it
  mistook a front rack for a lockout overhead, which is why triage frames are 768 px.
- **The judge cannot see the clip.** Left alone it returned confidence 0.85 on a clip that mixed a
  clean and a jerk. Every clip-quality cap in this README exists because the judge only ever
  reads text; the caps are ceilings on what it says, not a second opinion on the lift.
- **Reasoning models think instead of answering.** Without the no-thinking hint the judge spends
  the whole completion on reasoning, hits `finish_reason: length` and never emits the JSON; with
  it, a reply that still is not JSON gets one repair retry, then the fallback judge — and in the
  public demo, which runs without a fallback, an honest error.
- **The vision stage is not an NVIDIA model.** The Nemotron models on this account reject image
  input, so perception runs on MiniCPM-V; Nemotron does all the reasoning and scoring. The card
  says so on every verdict.

**Scope and platform**
- **One lift per clip works best.** A clean & jerk is detected and split, but the split relies on
  the athlete being briefly still between the two lifts.
- **Only snatch, clean and jerk.** Squats, deadlifts, presses and partial lifts are out of scope
  and are skipped rather than scored.
- **Some phone codecs will not play in the browser.** HEVC from Android Chrome, for one: the
  server reads it fine, but the page cannot trim it, so the clip is scored whole — and a public
  demo with a 15 s cap will refuse a long one. Trim it in the gallery app first. iOS Safari
  reports an infinite duration for a freshly captured clip until it seeks, which the trim screen
  works around.
- **No accounts, no history.** Results live for `JOB_TTL_SECONDS` and are then deleted. Nothing is
  stored between sessions.
- **Rotation needs ffprobe.** Without it, portrait clips may be judged sideways.
- **The public demo has a daily limit.** It is free and open, so it is bounded: 150 model calls
  a day for everyone (about 50 single-lift clips, fewer for clean & jerks), 12 an hour per
  address, one run at a time, 15 s and 60 MB per clip, two lifts per clip, a 120 s deadline per
  run, with the last fifth of the day kept for addresses that have not scored yet. Past a limit
  the page says so in one sentence and offers the pre-recorded sample result, which is the full
  experience at no cost. The live judge is Nemotron 3 Super, not Ultra, for the same reason.
- **Jobs and demo counters live in one process.** The store is in-memory, so the MVP runs as a
  single instance; **a restart resets the day's count and the per-address counters** (Heroku
  restarts dynos at least daily), which errs on the side of letting a judge in rather than
  locking one out, and it does not scale horizontally as-is.
- **Latency** is roughly a minute on a normal connection. That is a target observed in testing,
  not a guarantee — it depends on Token Factory load and clip length — and a run is cut off at
  `JOB_TIMEOUT_SECONDS` regardless.

### Deployment
The app is one container. The `Dockerfile` installs ffmpeg and the DejaVu font the replay strip
is drawn with, runs as a non-root user, answers `/healthz`, and **is the public demo**: the spend
caps are on in it (Super judge, no fallback, budgets in model calls with the day turning at
midnight in Israel, one run at a time, a 120 s run deadline, 15 s / 60 MB / two lifts per clip)
and the sample clip is baked in. Build the sample first, then the image:

```bash
python tools/make_demo_sample.py samples/snatch1.mp4 --lift snatch --out demo_sample/
docker build -t lift-judge .
docker run --rm -p 8000:8000 --env-file .env -e ACCESS_MODE=open lift-judge          # no code
docker run --rm -p 8000:8000 --env-file .env -e DEMO_ACCESS_CODE=the-code lift-judge # gated
```

That is enough for any container host (Fly.io, Render, Cloud Run, a Nebius AI Cloud VM…) that
terminates HTTPS for you — set `TRUSTED_PROXIES` to match the platform's proxy (`*` where
nothing but that proxy can reach the port) so the per-IP limit sees visitors rather than the
proxy. On a plain VM with a domain name, `deploy/` does the HTTPS too:

```bash
cp .env.example .env            # NEBIUS_API_KEY and DEMO_ACCESS_CODE at least
DOMAIN=lift.example.com docker compose -f deploy/docker-compose.yml up -d --build
```

Caddy obtains and renews the certificate, allows a 220 MB request body and 10-minute proxy
timeouts, and forwards to the app; the compose file sets `TRUSTED_PROXIES=private` for it, and
the app port is exposed to Caddy only, never published. Whatever you deploy on, it needs:
- `NEBIUS_API_KEY`, and either `ACCESS_MODE=open` or a `DEMO_ACCESS_CODE` (the other variables
  have defaults in the image)
- **HTTPS**, which camera capture and PWA install both require
- A reverse-proxy body limit at least as large as `MAX_UPLOAD_MB`, and generous read timeouts
- **One instance**: jobs and the demo counters are held in memory, so run a single replica (the
  image starts one uvicorn worker on purpose)

#### Heroku
The same image runs on Heroku's container stack. Log in once, then one script creates the app,
sets the config vars, builds the image locally, pushes and releases it on one dyno:

```bash
heroku login
APP=lift-judge-demo DEMO_ACCESS_CODE=the-code deploy/heroku.sh     # gated, code mode
APP=lift-judge-demo ACCESS_MODE=open deploy/heroku.sh              # open, caps on
```

The script reads `NEBIUS_API_KEY` from the environment or `.env` (use a key made just for the
demo), sets `TRUSTED_PROXIES=*`, `MAX_CONCURRENT_JOBS=1` and `MAX_UPLOAD_MB=60`, passes
`ACCESS_MODE` through when given, and prints the app's URL and the startup lines that list the
caps in force. Re-run it to redeploy. It refuses to deploy `open` with no sample built. `heroku.yml`
is there too, so `git push heroku main` also works, and carries the sample when `demo_sample/` is
committed; the script's local build carries it either way.

**Two build paths.** Heroku builds the Dockerfile only on the *container* stack
(`heroku stack:set container -a <app>`; `heroku.yml` points at it). An app on the default stack —
what `heroku create` without `--stack container` and the dashboard's GitHub connection give you —
builds with Heroku's **Python buildpack** instead; its log opens with *"Python app detected"*.
That path works too, with three files in the repo and one setting in the dashboard:
- `.python-version` pins 3.13. The buildpack's default is the newest Python (3.14), for which
  pydantic-core has no wheel, so the build tries to compile it and fails — the error you get
  without the pin.
- `Procfile` starts `deploy/heroku-web.sh`, which applies the same demo defaults the image has as
  `ENV` (Super judge, no fallback, budgets, the Israel-time day, deadline, clip, upload and
  lifts-per-clip caps), so the demo is capped the same way on both paths — and so `open` mode,
  which refuses to start without every one of them, starts — plus the Heroku-specific
  `TRUSTED_PROXIES=*`. A config var set in the dashboard always wins over a default, and
  `ACCESS_MODE` is only ever a config var.
- `Aptfile` installs ffmpeg, which the stack image lacks (it does have glib and the DejaVu font).
  It needs the apt buildpack **ahead of** the Python one: in the dashboard, *Settings →
  Buildpacks → Add* `heroku-community/apt` and drag it to the top, or
  `heroku buildpacks:add --index 1 heroku-community/apt -a <app>`. Without it the app runs, but
  with no annotated replay and portrait clips judged sideways; the startup log says so.

On either path set `NEBIUS_API_KEY` as a config var, plus `DEMO_ACCESS_CODE` unless
`ACCESS_MODE=open`. A build from git ships the sample clip only if `demo_sample/` is committed
(the sample is the one video the repo allows, see above); the script's local image build ships
it either way. The live app today runs the buildpack path, so the sample reaches it by commit.

**Going open, one variable at a time.** With the sample live and `/api/config` showing the caps:
`heroku config:set ACCESS_MODE=open -a <app>`. The dyno restarts, the first screen loses its
code field, and the startup line reads *access open*. `ACCESS_MODE=link` gates uploads again
behind the `?code=` link (and `/api/access/check` validates it on *Next*); `DEMO_DISABLED=true`
leaves the sample only. Each is a config change, never a deploy.

**"Application Error" on the URL** is Heroku's own page for an app with no dyno answering, and
`heroku logs --tail -a <app>` (or the dashboard's *More → View logs*) names the cause.
`H14 No web processes running` means the last build defined no web process — a buildpack build
before the `Procfile` existed — or the app is scaled to zero (`heroku ps:scale web=1`).
`H10 App crashed` means the process died on boot, and the lines above it say why. A failed build
leaves the previous release running or, on a new app, nothing at all.

What Heroku changes:
- **Uploads have 30 seconds.** The router gives every request a 30 s window to its first response
  byte, and the body transfer counts, so a clip that takes longer than that to upload fails with
  an H12 before the app sees it. The page's upload bar just stops. Film short clips, or trim in the
  gallery app before uploading; `MAX_UPLOAD_MB=60` makes the page warn from 30 MB. The 15 s window
  cap does not help here, because trimming happens on the server after the upload.
- **The router is the proxy.** Only it reaches the dyno, and it appends the visitor's address as
  the last `X-Forwarded-For` entry — exactly what `TRUSTED_PROXIES=*` reads, so the per-IP limit
  sees visitors and a forged header still buys nothing.
- **512 MB dynos.** OpenCV, a decoded clip and an ffmpeg render fit, but not twice over; keep
  `MAX_CONCURRENT_JOBS=1` on Eco, Basic and Standard-1X, or use a larger dyno. Eco dynos sleep
  after 30 minutes idle and the app takes ~10 s to wake and import OpenCV, so a judge's first tap
  waits; pick the cheapest dyno that does not sleep (`DYNO_TYPE=basic`, or `heroku ps:type`).
- **Dynos restart at least daily.** Jobs, replays and the demo counters live in the process, so a
  restart clears results that are still within their TTL and resets the day's budget and the
  per-address counters. One dyno only, as everywhere: never scale `web` above 1.
- **HTTPS is the router's**, so camera capture and PWA install work with no further setup.
  `heroku logs --tail -a <app>` shows the startup line and every refusal.

Then open the URL on a phone over 4G, score a clip, and check that the replay plays inline and
shares. The image was built and smoke-tested locally (`/healthz`, the page, the manifest,
ffmpeg, the font and OpenCV all present); the public URL is at the top of this README.

### Demo video
The finished video is linked at the top of this README. Its narration script, with per-beat
timings, shot directions and a pre-recording checklist, is in
[docs/demo-script.md](docs/demo-script.md).

### License
MIT (already present in this repository).

### Attribution
Biomechanics principles from Wingate Institute course material (Dr. Tavor Ben Ze'ev / Hoffman Lab), used as principles only.

