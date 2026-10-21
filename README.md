## Sideline Lift Judge

Film one Olympic lift from the side on your phone and get back a **1–10 bar-path efficiency
score**, a band, the **maximum horizontal bar drift**, a **confidence** value, notes and **1–2
coaching cues** — plus an annotated replay you can download or share. Mobile-first, so a coach
can use it from the side of the platform.

Built for the Nebius × NVIDIA Global AI Hackathon. The judge is an **NVIDIA Nemotron 3 Ultra**
model, and both stages run on **Nebius Token Factory**.

- Mobile-first web app, installable as a PWA — film or upload, trim, score, share
- Two kinds of evidence, one judge: a local plate tracker **measures** the bar path and a vision
  model **describes** it; NVIDIA Nemotron weighs both against the rubric
- Clean & jerk is detected and scored as **two separate lifts**
- Confidence drops, with a visible warning, on portrait, off-axis or mixed-lift clips — and
  when the judge's verdict contradicts the measured path
- Schema-validated JSON, a repair retry, and a fallback judge model
- Two CLIs for one clip and for a whole session

### How a clip becomes a verdict

```
 phone clip ──► trim ──► plate tracker (OpenCV, local, free)
                            │  finds each lift: where the plate rises
                            │  follows the near plate through it
                            ├─► bar-path MEASUREMENT  (drift in plate diameters,
                            │   where in the rise, coverage)          ──────────┐
                            ▼                                                   │
                      8–16 frames from the lift window                          │
                            │                                                   │
                            ▼                                                   ▼
             Stage 1  MiniCPM-V (Token Factory)              Stage 2  NVIDIA Nemotron 3 Ultra
             "what the bar did", phase by phase   ────────►  rubric + observations + measurement
             direction, bar–body contact, angle              → validated JSON: score, band, drift,
                                                               cues, confidence
                                                                        │
                   confidence caps (local): portrait, off-axis, mixed   │
                   lifts, no trace, verdict vs. measurement  ◄──────────┘
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
  utils/             bar_track.py (plate tracker: finds lifts, draws the trace),
                     frames, rotation/probing, segmentation, clip cutting, uploads
client/              everything served to the browser — no Python
  templates/         index.html (Jinja)
  static/            styles.css, app.js, service-worker.js, manifest.webmanifest,
                     img/ (mark, filming guide), icons/ (PWA icons)
prompts/             model prompts, shared by the server and both CLIs
tools/               make_icons.py — regenerates the PWA icons
tests/               pytest; every Token Factory call is mocked
score_clip.py        CLI: one clip
score_session.py     CLI: a whole session, split into exercises
Dockerfile           the app as one container (ffmpeg and fonts included)
deploy/              docker-compose.yml + Caddyfile: HTTPS on a VM in one command
```

### Nebius Token Factory models (two-stage)
- Base URL: `https://api.tokenfactory.nebius.com/v1`
- API key: `NEBIUS_API_KEY` (required)
- Stage 1 (vision/perception): `openbmb/MiniCPM-V-4_5` (configurable via `NEBIUS_VISION_MODEL`) — receives sampled frames and outputs factual observations only (no scores).
- Stage 2 (judge/reasoning): `nvidia/Nemotron-3-Ultra-550b-a55b` (configurable via `NEBIUS_JUDGE_MODEL`) — receives the rubric + per-lift prompt and the Stage‑1 observations, and returns the validated JSON score.
- Judge fallback: `nvidia/nemotron-3-super-120b-a12b` (configurable via `NEBIUS_JUDGE_FALLBACK_MODEL`). If the primary judge errors, times out, or never returns valid JSON — even after the one repair retry — the same prompt is sent to this model. Set the variable to an empty value to turn the fallback off.
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
- Public-demo caps, all off until set — see [Running the public demo](#running-the-public-demo-without-a-surprise-bill):
  `DEMO_ACCESS_CODE`, `DEMO_DAILY_LIMIT`, `DEMO_IP_HOURLY_LIMIT`, `MAX_CONCURRENT_JOBS`,
  `MAX_CLIP_SECONDS`, `MAX_FRAMES`, `DEMO_SAMPLE_DIR`

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

The result screen shows the score and band in large type, the max horizontal drift, a confidence
meter with any warnings next to it, 1–2 cue cards, collapsible notes, and the annotated replay
with download and share buttons.

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

Jobs and their replays are deleted after `JOB_TTL_SECONDS` (default 30 minutes), and the uploaded
clip is deleted as soon as scoring finishes.

### Filming guide
The score is only as good as the angle. The app shows this before the camera opens, and it is
worth repeating here:

- Film **square to the side** — not in front, not at an angle. Drift is measured sideways.
- **3–5 m away**, camera at **hip height**.
- **Landscape** preferred. Portrait works but lowers the confidence value.
- Keep the **whole bar and both plates in frame** for the entire lift.
- **One lift per clip.** Trim it on the trim screen; a clean & jerk is scored as two lifts.

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
| Filmed in portrait | 0.60 |
| Phases not clearly separable | 0.60 |
| Clip too short to hold a lift | 0.60 |
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
result says which you got.

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
`bar_path` and are shown under the drift description as *Tracked: bar strayed 0.21 plate widths…*.
When the bar could not be followed, the judge is told that too and is asked to keep its confidence
down, and the server caps it anyway.

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
after it), and no window is allowed to be longer than one lift. The motion-energy segmentation
remains only as a fallback when nothing could be tracked.

**Measured on 39 real clips** (62 hand-labelled lifts, `tests/data/lift_windows_truth.json`,
re-measured 2026-10-21): recall 85% (9 lifts missed), precision 78% (11 junk windows of 51).
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

**Then limit usage in the app.** Every cap below is off until its variable is set, so local
runs, the CLIs and the tests are untouched. A refused request gets one plain sentence, a
`Retry-After` header and a machine-readable `reason` the page acts on.

| Layer | Variable | Suggested | What the visitor sees |
|---|---|---|---|
| Access code | `DEMO_ACCESS_CODE` | a code only in the submission | a field on the first screen; `?code=…` in the link fills it and is then removed from the address bar |
| Daily budget | `DEMO_DAILY_LIMIT` | `50` | "Today's demo budget … is used up" (HTTP 429, resets at UTC midnight) |
| Per-IP rate | `DEMO_IP_HOURLY_LIMIT` | `3` | "You've scored 3 clips in the last hour … try again in N min" |
| In flight at once | `MAX_CONCURRENT_JOBS` | `2` | "scoring other clips right now … try again in a minute" |
| Clip length | `MAX_CLIP_SECONDS` | `15` | the end handle starts at 15 s and the score button waits until the window fits; the server refuses anything longer |
| Frames per lift | `MAX_FRAMES` | `10` | nothing — one vision call per lift instead of two |
| Cheaper judge | `NEBIUS_JUDGE_MODEL` | `nvidia/nemotron-3-super-120b-a12b` | the footer names the judge that is actually configured |

The cap is on the *scored window*, not the upload: a long gallery clip trimmed to one lift is
fine. Several access codes can be comma-separated, so one can be revoked without the rest.
Counters live in the process, like the job store, so a restart resets the day's count — which
errs on the side of letting a judge in. The per-IP limit reads the first `X-Forwarded-For`
entry, which a direct caller could forge; all that buys them is one address's share, and the
daily budget still holds. The sample, the polls and the replays are never counted.

**Have a fallback that costs nothing.** Score one clip once, ahead of time, and the app serves
that result — replay included — from disk:

```bash
python tools/make_demo_sample.py samples/snatch1.mp4 --lift snatch --out demo_sample/
# then on the server:
DEMO_SAMPLE_DIR=demo_sample
```

A **Try a sample clip** button appears on the filming screen, and again next to every refusal,
so a judge who arrives after the budget is spent still sees the whole result screen. The sample
needs no access code and is not counted. Pick a clip you are allowed to show in public; the
directory holds video, so it is gitignored and deployed alongside the app, never committed.

**Keep Ultra for your own recorded run.** The public demo can run the cheaper Nemotron 3 Super
as its judge while the default — and the video you record locally — stays on Ultra. The page
footer always names the judge that is actually configured, so the demo never claims a model it
is not using.

### Upload limits and errors
- Uploads are streamed to disk and rejected over `MAX_UPLOAD_MB` (HTTP 413) with a message
  asking you to trim the clip; the partial file is removed.
- Any `video/*` type is accepted, plus generic types (e.g. `application/octet-stream`) that
  carry a video extension — phones are inconsistent about what they send.
- Any failure reaches the phone as one plain sentence that says what to do next ("The model
  service is busy right now. Wait a minute and try the clip again."), never the provider's
  reply or a stack trace; those go to the server log, and to `error_detail` for the CLIs
  (`server/services/errors.py`).
- Token Factory calls connect within 10 s and must answer within `NEBIUS_TIMEOUT_SECONDS`
  (default 45; a healthy call takes 2–3 s). A 429 or 5xx gets exactly one more try after a short
  pause; the judge then falls over to its second model. Dropping the no-thinking hint and
  asking again happens only when the provider rejects the parameter (400/422), not on an outage.
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
- Token estimate: depends on provider accounting for multi-image inputs; exact tokenization is provider-specific and not documented here. Treat as unknown and verify in your Nebius Token Factory usage dashboard.
- For a URL that is open to the public, cap the spend — see
  [Running the public demo](#running-the-public-demo-without-a-surprise-bill).

### Limitations
Read these before trusting a number.

- **Scores are indicative.** The judge reads a vision model's description of 8–16 sampled frames
  plus a drift figure tracked from one plate's apparent position. That figure is scaled by the
  plate's size, not calibrated against the scene, so parallax and an off-axis camera leak into
  it; there is no pose estimation. Treat the score as a coach's first read, not a lab result.
- **Angle decides everything.** A clip that is not square to the side cannot show horizontal drift
  honestly. The confidence value reflects this, but it cannot rescue a bad angle.
- **One lift per clip works best.** A clean & jerk is detected and split, but the split relies on
  the athlete being briefly still between the two lifts.
- **Only snatch, clean and jerk.** Squats, deadlifts, presses and partial lifts are out of scope
  and are skipped rather than scored.
- **No accounts, no history.** Results live for `JOB_TTL_SECONDS` and are then deleted. Nothing is
  stored between sessions.
- **Rotation needs ffprobe.** Without it, portrait clips may be judged sideways.
- **Jobs live in one process.** The store is in-memory, so the MVP runs as a single instance; it
  does not scale horizontally as-is.
- **Latency** is roughly a minute on a normal connection. That is a target observed in testing,
  not a guarantee — it depends on Token Factory load and clip length.

### Deployment
The app is one container. The `Dockerfile` installs ffmpeg and the DejaVu font the replay strip
is drawn with, runs as a non-root user and answers `/healthz`:

```bash
docker build -t lift-judge .
docker run --rm -p 8000:8000 --env-file .env lift-judge
```

That is enough for any container host (Fly.io, Render, Cloud Run, a Nebius AI Cloud VM…) that
terminates HTTPS for you. On a plain VM with a domain name, `deploy/` does the HTTPS too:

```bash
cp .env.example .env            # NEBIUS_API_KEY, plus the public-demo caps above
DOMAIN=lift.example.com docker compose -f deploy/docker-compose.yml up -d --build
```

Caddy obtains and renews the certificate, allows a 220 MB request body and 10-minute proxy
timeouts, and forwards to the app. Whatever you deploy on, it needs:
- `NEBIUS_API_KEY` (and optionally the other variables above)
- **HTTPS**, which camera capture and PWA install both require
- A reverse-proxy body limit at least as large as `MAX_UPLOAD_MB`, and generous read timeouts
- **One instance**: jobs and the demo counters are held in memory, so run a single replica (the
  image starts one uvicorn worker on purpose)

Then open the URL on a phone over 4G, score a clip, and check that the replay plays inline and
shares. The image was built and smoke-tested locally (`/healthz`, the page, the manifest,
ffmpeg, the font and OpenCV all present); the public URL itself is not part of this repository.

### Demo video
The narration script for the ≤3-minute submission video, with per-beat timings, shot directions
and a pre-recording checklist, is in [docs/demo-script.md](docs/demo-script.md).

### License
MIT (already present in this repository).

### Attribution
Biomechanics principles from Wingate Institute course material (Dr. Tavor Ben Ze'ev / Hoffman Lab), used as principles only.

