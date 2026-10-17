## Sideline Lift Judge — MVP

First working MVP for the Nebius x NVIDIA Global AI Hackathon.

- Simple upload UI (FastAPI) + CLI
- Evenly samples ~8–16 frames and runs a two-stage pipeline on Nebius Token Factory (OpenAI-compatible)
- Validates the model’s JSON output against a strict schema and retries once on parse failure

### Stack
- FastAPI + Jinja2 (single-page upload UI)
- OpenCV for frame extraction
- OpenAI-compatible HTTP calls to Nebius Token Factory
- Pydantic for JSON schema validation
- Pytest for unit tests (frame sampling, JSON validation, API call mocked)

### Nebius Token Factory models (two-stage)
- Base URL: `https://api.tokenfactory.nebius.com/v1`
- API key: `NEBIUS_API_KEY` (required)
- Stage 1 (vision/perception): `openbmb/MiniCPM-V-4_5` (configurable via `NEBIUS_VISION_MODEL`) — receives sampled frames and outputs factual observations only (no scores).
- Stage 2 (judge/reasoning): `nvidia/Nemotron-3-Ultra-550b-a55b` (configurable via `NEBIUS_JUDGE_MODEL`) — receives the rubric + per-lift prompt and the Stage‑1 observations, and returns the validated JSON score.
- Judge fallback: `nvidia/nemotron-3-super-120b-a12b` (configurable via `NEBIUS_JUDGE_FALLBACK_MODEL`). If the primary judge errors, times out, or never returns valid JSON — even after the one repair retry — the same prompt is sent to this model. Set the variable to an empty value to turn the fallback off.\n- Cosmos models are not currently served on Token Factory in this account. If needed later, Cosmos on AI Cloud is a future option outside Token Factory.

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
1) Python 3.11+ recommended (tested on 3.12)
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
```bash
uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --reload
```
Open `http://localhost:8000`:
- Choose lift type (snatch / clean / jerk)
- Upload a short side-view clip
- The app extracts ~10 frames by default (clamped to 8–16), auto-rotates based on phone metadata (when available) and downsizes frames to ~768px long side, then:
  - Stage 1 (MiniCPM‑V): produces plain-English observations only.
  - Stage 2 (Nemotron): produces the final JSON (score 1–10, band, max horizontal drift, 1–2 cues, confidence, optional notes). If Stage‑1 notes occlusion/oblique angle, confidence is lowered accordingly.

### CLI (no UI)
```bash
python score_clip.py /path/to/clip.mp4 --lift snatch
```
Options:
- `--frames 10` (clamped to 8–16; at most 10 images per vision call; extra frames are chunked)
- `--prompts_dir prompts/`
- `--debug` (prints Stage‑1 observations)

Outputs the validated JSON to stdout. Retries once if the first response is invalid JSON,
then falls back to the secondary judge model if the primary one still fails.

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

### Upload limits and errors
- Uploads are streamed to disk and rejected over `MAX_UPLOAD_MB` (HTTP 413) with a message
  asking you to trim the clip; the partial file is removed.
- Any `video/*` type is accepted, plus generic types (e.g. `application/octet-stream`) that
  carry a video extension — phones are inconsistent about what they send.
- Upstream model failures return a single plain sentence (HTTP 502/503). Stack traces go to
  the server log only.
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

### Deployment
The app is deliberately small to ease later deployment (container, serverless, or a managed app platform). It requires only:
- `NEBIUS_API_KEY` (and optionally `NEBIUS_BASE_URL`, `NEBIUS_VISION_MODEL`, `NEBIUS_JUDGE_MODEL`)
- A Python runtime

### License
MIT (already present in this repository).

### Attribution
Biomechanics principles from Wingate Institute course material (Dr. Tavor Ben Ze'ev / Hoffman Lab), used as principles only.

