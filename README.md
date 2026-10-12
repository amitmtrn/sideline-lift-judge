## Sideline Lift Judge — MVP

First working MVP for the Nebius x NVIDIA Global AI Hackathon.

- Simple upload UI (FastAPI) + CLI
- Evenly samples ~8–16 frames, sends them as images to an NVIDIA open vision model via Nebius Token Factory (OpenAI-compatible)
- Validates the model’s JSON output against a strict schema and retries once on parse failure

### Stack
- FastAPI + Jinja2 (single-page upload UI)
- OpenCV for frame extraction
- OpenAI-compatible HTTP calls to Nebius Token Factory
- Pydantic for JSON schema validation
- Pytest for unit tests (frame sampling, JSON validation, API call mocked)

### NVIDIA model on Nebius Token Factory
- Base URL: `https://api.tokenfactory.nebius.com/v1`
- API key: `NEBIUS_API_KEY` (required)
- Default model id (configurable via `NEBIUS_MODEL`): `nvidia/Cosmos3-Super-Reasoner`
  - Why: Cosmos 3 Super Reasoner is a general-purpose VLM suitable for multi-image physical reasoning from side-view lifting clips.
  - Verified against Nebius Token Factory docs:
    - Models listing: see `GET /v1/models` in docs (`docs.tokenfactory.nebius.com/api-reference/models/list-models`)
    - Example OpenAI SDK usage and base URL: `https://api.tokenfactory.nebius.com/v1/`
  - What is still unverified here: live inference in this repo (no key included). The code path and mocks are in place; run locally with your key to verify end-to-end.

You can confirm availability in your workspace:

```bash
curl -H "Authorization: Bearer $NEBIUS_API_KEY" https://api.tokenfactory.nebius.com/v1/models | jq .
```

If your account exposes a different Cosmos id (e.g. `nvidia/Cosmos3-Super`), set:

```bash
export NEBIUS_MODEL="nvidia/Cosmos3-Super"
```

### Prompts
The prompts are fixed, stored under `prompts/`:
- `prompts/system.txt` (physics/rubric; Wingate principles)
- `prompts/user_snatch.txt`, `prompts/user_clean.txt`, `prompts/user_jerk.txt`

### Setup
1) Python 3.11+ recommended (tested on 3.12)
2) Create `.env` (see `.env.example`) with at least:
   - `NEBIUS_API_KEY=...`
   - Optionally override `NEBIUS_BASE_URL` and `NEBIUS_MODEL`
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
- The app extracts ~12 frames (clamped to 8–16), calls the model, validates output JSON, and displays score (1–10), band, max horizontal drift, 1–2 coaching cues, confidence, and the raw JSON.

### CLI (no UI)
```bash
python score_clip.py /path/to/clip.mp4 --lift snatch
```
Options:
- `--frames 12` (clamped to 8–16)
- `--prompts_dir prompts/`

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
- Nebius API call: request layer is mocked to validate response parsing (no live inference)

### Cost note
- Frames per call: by default 12 images are sent (configurable to 8–16).
- Token estimate: depends on provider accounting for multi-image inputs; exact tokenization is provider-specific and not documented here. Treat as unknown and verify in your Nebius Token Factory usage dashboard.

### Deployment
The app is deliberately small to ease later deployment (container, serverless, or a managed app platform). It requires only:
- `NEBIUS_API_KEY` (and optionally `NEBIUS_MODEL`, `NEBIUS_BASE_URL`)
- A Python runtime

### License
MIT (already present in this repository).

### Attribution
Biomechanics principles from Wingate Institute course material (Dr. Tavor Ben Ze'ev / Hoffman Lab), used as principles only.

