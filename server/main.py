import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from server.schemas.judgment import LiftType
from server.services.demo_guard import (
    DemoRefused,
    access_required,
    check_access,
    client_ip,
    enforce_clip_length,
    get_guard,
    max_clip_seconds,
)
from server.services.errors import UserFacingError
from server.services.jobs import Job, get_store, job_ttl_seconds
from server.services.nebius_client import NebiusChatClient, judge_display_name
from server.services.pipeline import (
    DEFAULT_FRAMES,
    MIN_FRAMES,
    ScoreRequest,
    TrimError,
    clamp_frames,
    max_frames,
    resolve_trim,
    run_scoring,
)
from server.services.results import lift_payload, outcome_payload
from server.services.sample import load_sample
from server.utils.uploads import (
    UnsupportedUpload,
    UploadTooLarge,
    is_supported_video_upload,
    max_upload_bytes,
    save_upload_to_temp,
)
from server.utils.video_meta import probe_video

# Load .env if present for local dev
load_dotenv()

SERVER_DIR = Path(__file__).resolve().parent
BASE_DIR = SERVER_DIR.parent
PROMPTS_DIR = BASE_DIR / "prompts"
# The front end lives in client/ and is served from here; nothing in it is
# imported as Python, so the two halves stay separable.
CLIENT_DIR = BASE_DIR / "client"
TEMPLATES_DIR = CLIENT_DIR / "templates"
STATIC_DIR = CLIENT_DIR / "static"

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    # Never leave uploaded clips or replays behind (spec §12).
    get_store().shutdown()


app = FastAPI(title="Sideline Lift Judge", lifespan=lifespan)

# Mount static and templates
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

LIFT_CHOICES = [lift.value for lift in LiftType]


class ConfigurationError(RuntimeError):
    """Server-side misconfiguration; surfaced as a plain message, never a stack trace."""


@app.exception_handler(DemoRefused)
async def demo_refused(_: Request, exc: DemoRefused) -> JSONResponse:
    """
    A demo limit, as one plain sentence plus a reason the page can act on
    (offer the sample clip, ask for the access code) and a Retry-After.
    """
    headers = {"Retry-After": str(exc.retry_after)} if exc.retry_after else None
    return JSONResponse(
        status_code=exc.status,
        content={"detail": exc.detail, "reason": exc.reason},
        headers=headers,
    )


def get_client() -> NebiusChatClient:
    api_key = os.getenv("NEBIUS_API_KEY")
    if not api_key:
        raise ConfigurationError("The scoring service is not configured yet. Set NEBIUS_API_KEY on the server.")
    base_url = os.getenv("NEBIUS_BASE_URL") or NebiusChatClient.DEFAULT_BASE_URL
    return NebiusChatClient(api_key=api_key, base_url=base_url)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "default_lift": "snatch",
            "lifts": LIFT_CHOICES,
            "min_frames": MIN_FRAMES,
            "max_frames": max_frames(),
            "default_frames": DEFAULT_FRAMES,
            "max_upload_mb": max_upload_bytes() // (1024 * 1024),
            "max_clip_seconds": max_clip_seconds(),
            "access_required": access_required(),
            "sample_available": load_sample() is not None,
            "judge_name": judge_display_name(),
        },
    )


@app.get("/manifest.webmanifest", include_in_schema=False)
async def manifest() -> FileResponse:
    # Served from the root so the PWA scope covers the whole app.
    return FileResponse(STATIC_DIR / "manifest.webmanifest", media_type="application/manifest+json")


@app.get("/service-worker.js", include_in_schema=False)
async def service_worker() -> FileResponse:
    # A worker can only control pages at or below its own path, so this must not
    # live under /static.
    return FileResponse(
        STATIC_DIR / "service-worker.js",
        media_type="application/javascript",
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> FileResponse:
    return FileResponse(STATIC_DIR / "icons" / "icon-192.png", media_type="image/png")


def _resolve_upload(video: Optional[UploadFile], file: Optional[UploadFile]) -> UploadFile:
    """Spec §10 names the field `video`; `file` is kept for the original API."""
    upload = video or file
    if upload is None or not getattr(upload, "filename", None):
        raise HTTPException(status_code=400, detail="No clip was uploaded.")
    return upload


def _parse_optional_float(value: Optional[str], label: str) -> Optional[float]:
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail=f"{label} must be a number of seconds.")


def _parse_lift(value: Optional[str]) -> Optional[LiftType]:
    if value is None or str(value).strip() == "" or str(value).strip().lower() == "auto":
        return None
    try:
        return LiftType(str(value).strip().lower())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown lift: {value}. Choose snatch, clean or jerk.")


def _access_code(request: Request, form_value: Optional[str]) -> Optional[str]:
    """The page sends the code as a header; scripts may send it as a form field."""
    return request.headers.get("x-access-code") or form_value


async def _store_upload(upload: UploadFile, destination_dir: Optional[Path] = None) -> Path:
    if not is_supported_video_upload(upload.content_type, upload.filename):
        raise HTTPException(
            status_code=400,
            detail="That file is not a video. Upload a clip filmed on your phone (MP4 or MOV).",
        )
    try:
        return await save_upload_to_temp(
            upload,
            max_bytes=max_upload_bytes(),
            dir=str(destination_dir) if destination_dir else None,
        )
    except UploadTooLarge as exc:
        raise HTTPException(status_code=413, detail=str(exc))
    except UnsupportedUpload as exc:
        raise HTTPException(status_code=400, detail=str(exc))


def _check_window(source: Path, start: Optional[float], end: Optional[float]) -> None:
    """
    Validate the trim window against the clip now, so the user hears about it
    immediately, and hold it to the demo's clip-length cap.
    """
    info = probe_video(str(source))
    try:
        window_start, window_end, _ = resolve_trim(duration_sec=info.duration_sec, trim_start=start, trim_end=end)
    except TrimError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    enforce_clip_length(window_start, window_end)


@app.post("/api/jobs")
async def create_job(
    request: Request,
    video: Optional[UploadFile] = File(default=None),
    file: Optional[UploadFile] = File(default=None),
    lift: Optional[str] = Form(default=None),
    mode: str = Form(default="single"),
    trim_start: Optional[str] = Form(default=None),
    trim_end: Optional[str] = Form(default=None),
    frames: Optional[int] = Form(default=None),
    num_frames: Optional[int] = Form(default=None),
    debug: bool = Form(default=False),
    access_code: Optional[str] = Form(default=None),
):
    """
    Accept a clip and start scoring in the background (spec §13). Returns the job
    id to poll, so a slow or dropped mobile connection cannot lose the result.
    """
    check_access(_access_code(request, access_code))
    upload = _resolve_upload(video, file)
    selected_lift = _parse_lift(lift)
    start = _parse_optional_float(trim_start, "Trim start")
    end = _parse_optional_float(trim_end, "Trim end")
    mode = "auto" if str(mode).strip().lower() == "auto" else "single"

    # Fail before touching the network if the service is not configured.
    try:
        client = get_client()
    except ConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    store = get_store()
    guard = get_guard()
    job = store.create()
    try:
        source = await _store_upload(upload, job.workdir)
        job.source_path = source
        _check_window(source, start, end)
        # Everything free has passed; only now does the clip count against the
        # caller's allowance, so a rejected window never costs them a turn.
        guard.admit(client_ip(request))
    except (HTTPException, DemoRefused):
        store.discard(job)
        raise

    request_model = ScoreRequest(
        video_path=str(source),
        lift=selected_lift,
        trim_start=start,
        trim_end=end,
        frames=clamp_frames(frames if frames is not None else num_frames),
        mode=mode,
        render_replay=True,
        replay_dir=str(job.workdir),
        keep_observations=bool(debug),
    )

    async def runner(current: Job) -> None:
        try:
            outcome = await run_scoring(
                request_model,
                client=client,
                prompts_dir=str(PROMPTS_DIR),
                progress=current.set_progress,
            )
            current.replays = {
                item.index: Path(item.replay_path) for item in outcome.lifts if item.replay_path
            }
            current.payload = outcome_payload(
                outcome, replay_url=lambda index: f"/api/jobs/{current.id}/replay/{index}.mp4"
            )
            if not outcome.scored:
                first_error = next((item.error for item in outcome.lifts if item.error), None)
                if first_error:
                    raise UserFacingError(first_error)
                current.set_progress("done", "No snatch, clean or jerk found in this clip.", 1.0)
                return
            current.set_progress("done", "Done.", 1.0)
        finally:
            guard.release()

    store.start(job, runner)
    return JSONResponse(status_code=202, content={"id": job.id, "poll_url": f"/api/jobs/{job.id}"})


@app.get("/api/jobs/{job_id}")
async def read_job(job_id: str):
    job = get_store().get(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail=f"That result has expired. Results are kept for {job_ttl_seconds() // 60} minutes.",
        )
    return JSONResponse(content=job.as_dict())


@app.get("/api/jobs/{job_id}/replay/{index}.mp4")
async def read_replay(job_id: str, index: int):
    job = get_store().get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="That replay has expired.")
    path = job.replays.get(index)
    if path is None or not Path(path).is_file():
        raise HTTPException(status_code=404, detail="No replay for that lift.")
    # Inline so iOS Safari plays it in the page; the download button adds its own
    # filename via the anchor.
    return FileResponse(
        path,
        media_type="video/mp4",
        headers={"Content-Disposition": f'inline; filename="lift-{index}.mp4"'},
    )


@app.get("/api/sample")
async def read_sample():
    """
    The cached, pre-scored sample clip: free to serve, no access code, not
    counted against any limit. What a judge sees when the budget is spent.
    """
    sample = load_sample()
    if sample is None:
        raise HTTPException(status_code=404, detail="No sample clip is set up on this server.")
    return JSONResponse(content=sample.payload)


@app.get("/api/sample/replay/{index}.mp4")
async def read_sample_replay(index: int):
    sample = load_sample()
    path = sample.replays.get(index) if sample is not None else None
    if path is None:
        raise HTTPException(status_code=404, detail="No replay for that sample lift.")
    return FileResponse(
        path,
        media_type="video/mp4",
        headers={
            "Content-Disposition": f'inline; filename="sample-lift-{index}.mp4"',
            # The same file every time: let the phone keep it.
            "Cache-Control": "public, max-age=3600",
        },
    )


@app.post("/api/score")
async def score_endpoint(
    request: Request,
    video: Optional[UploadFile] = File(default=None),
    file: Optional[UploadFile] = File(default=None),
    lift: Optional[str] = Form(default=None),
    mode: str = Form(default="single"),
    trim_start: Optional[str] = Form(default=None),
    trim_end: Optional[str] = Form(default=None),
    frames: Optional[int] = Form(default=None),
    num_frames: Optional[int] = Form(default=None),
    debug: bool = Form(default=False),
    access_code: Optional[str] = Form(default=None),
):
    """
    Synchronous scoring: one request, one verdict. Kept for scripts and for
    parity with the CLI; the web UI uses /api/jobs so it can show progress.
    """
    check_access(_access_code(request, access_code))
    upload = _resolve_upload(video, file)
    selected_lift = _parse_lift(lift)
    start = _parse_optional_float(trim_start, "Trim start")
    end = _parse_optional_float(trim_end, "Trim end")

    source = await _store_upload(upload)
    guard = get_guard()
    admitted = False
    try:
        client = get_client()
        _check_window(source, start, end)
        guard.admit(client_ip(request))
        admitted = True
        outcome = await run_scoring(
            ScoreRequest(
                video_path=str(source),
                lift=selected_lift,
                trim_start=start,
                trim_end=end,
                frames=clamp_frames(frames if frames is not None else num_frames),
                mode="auto" if str(mode).strip().lower() == "auto" else "single",
                render_replay=False,
                keep_observations=bool(debug),
            ),
            client=client,
            prompts_dir=str(PROMPTS_DIR),
        )
        scored = outcome.scored
        if not scored:
            first_error = next((item.error for item in outcome.lifts if item.error), None)
            raise HTTPException(
                status_code=422,
                detail=first_error or "No snatch, clean or jerk could be scored in this clip.",
            )
        payload = outcome_payload(outcome)
        # One lift stays backwards compatible: the verdict is the top-level body.
        if len(scored) == 1 and len(outcome.lifts) == 1:
            body = lift_payload(scored[0])
            body["split_detected"] = outcome.split_detected
            return JSONResponse(content=body)
        return JSONResponse(content=payload)
    except (HTTPException, DemoRefused):
        raise
    except TrimError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except ConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.exception("Scoring failed")
        raise HTTPException(
            status_code=502,
            detail="Scoring failed. The model service may be busy — wait a moment and try the clip again.",
        ) from exc
    finally:
        if admitted:
            guard.release()
        Path(source).unlink(missing_ok=True)


@app.get("/healthz")
async def healthz():
    return {"ok": True}
