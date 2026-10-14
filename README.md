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
- Stage 2 (judge/reasoning): `nvidia/Nemotron-3-Ultra-550b-a55b` (configurable via `NEBIUS_JUDGE_MODEL`, with `nvidia/nemotron-3-super-120b-a12b` as an allowed fallback) — receives the rubric + per-lift prompt and the Stage‑1 observations, and returns the validated JSON score.\n- Cosmos models are not currently served on Token Factory in this account. If needed later, Cosmos on AI Cloud is a future option outside Token Factory.

Notes:
- Models listing: use `GET /v1/models` to verify availability in your workspace.
- Code paths are mocked in tests; run locally with your key to verify end‑to‑end.

You can confirm availability in your workspace:

```bash
curl -H "Authorization: Bearer $NEBIUS_API_KEY" https://api.tokenfactory.nebius.com/v1/models | jq .
```

Environment variables:
- `NEBIUS_VISION_MODEL` (default `openbmb/MiniCPM-V-4_5`)
- `NEBIUS_JUDGE_MODEL` (default `nvidia/Nemotron-3-Ultra-550b-a55b`)

### Prompts
The prompts are fixed, stored under `prompts/`:
- `prompts/system.txt` (physics/rubric; Wingate principles)
- `prompts/user_snatch.txt`, `prompts/user_clean.txt`, `prompts/user_jerk.txt`
- `prompts/vision_observation.txt` (Stage‑1 factual observation instructions)

### Setup
1) Python 3.11+ recommended (tested on 3.12)
2) Create `.env` (see `.env.example`) with at least:
   - `NEBIUS_API_KEY=...`
   - Optionally override `NEBIUS_BASE_URL`, `NEBIUS_VISION_MODEL`, `NEBIUS_JUDGE_MODEL`
3) Install deps:
   ```bash
   pip install -r requirements.txt
   ```

### Run the web app
```bash
uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --reload
```
Open `http://localhost:8000`:
- Choose lift type (snatch / clean / jerk)
- Upload a short side-view clip
- The app extracts ~12 frames (clamped to 8–16), auto-rotates based on phone metadata (when available) and downsizes frames to ~768px long side, then:
  - Stage 1 (MiniCPM‑V): produces plain-English observations only.
  - Stage 2 (Nemotron): produces the final JSON (score 1–10, band, max horizontal drift, 1–2 cues, confidence, optional notes). If Stage‑1 notes occlusion/oblique angle, confidence is lowered accordingly.

### CLI (no UI)
```bash
python score_clip.py /path/to/clip.mp4 --lift snatch
```
Options:
- `--frames 12` (clamped to 8–16)
- `--prompts_dir prompts/`
- `--debug` (prints Stage‑1 observations)

Outputs the validated JSON to stdout. Retries once if the first response is invalid JSON.

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

### Cost note
- Frames per call: by default 12 images are sent (configurable to 8–16).
- Token estimate: depends on provider accounting for multi-image inputs; exact tokenization is provider-specific and not documented here. Treat as unknown and verify in your Nebius Token Factory usage dashboard.

### Deployment
The app is deliberately small to ease later deployment (container, serverless, or a managed app platform). It requires only:
- `NEBIUS_API_KEY` (and optionally `NEBIUS_BASE_URL`, `NEBIUS_VISION_MODEL`, `NEBIUS_JUDGE_MODEL`)
- A Python runtime

### License
MIT (already present in this repository).

### Attribution
Biomechanics principles from Wingate Institute course material (Dr. Tavor Ben Ze'ev / Hoffman Lab), used as principles only.

