import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional, TypeVar

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from server.schemas.judgment import LiftType
from server.services.demo_guard import (
    Admission,
    DemoRefused,
    access_mode,
    access_required,
    check_access,
    check_enabled,
    client_ip,
    daily_limit,
    day_timezone,
    demo_caps_set,
    demo_disabled,
    enforce_clip_length,
    get_guard,
    ip_hash,
    ip_hourly_limit,
    job_timeout_seconds,
    max_clip_seconds,
    max_concurrent_jobs,
    max_lifts_per_clip,
    missing_open_mode_caps,
    trusted_proxies,
)
from server.services.errors import TOO_LONG, UserFacingError
from server.services.jobs import Job, get_store, job_ttl_seconds
from server.services.nebius_client import (
    NebiusChatClient,
    judge_display_name,
    judge_fallback_model,
    judge_model,
    model_display_name,
    vision_model,
)
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
from server.services.sample import load_sample, sample_dir
from server.utils.uploads import (
    UnsupportedUpload,
    UploadTooLarge,
    is_supported_video_upload,
    max_upload_bytes,
    save_upload_to_temp,
)
from server.utils.ffmpeg_binary import ffmpeg_exe, ffprobe_exe, log_binaries
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

# uvicorn configures only its own loggers. The app's go to the root logger,
# which left alone drops everything below WARNING — including the startup
# line that says which judge and caps are in force. LOG_LEVEL overrides.
if not logging.getLogger().handlers:
    logging.basicConfig(level=os.getenv("LOG_LEVEL") or "INFO", format="%(levelname)s:     %(name)s: %(message)s")

# The routes that cost model calls. The demo's gate runs for these before the
# request body is read (see DemoGateMiddleware).
GATED_PATHS = frozenset({"/api/jobs", "/api/score"})
# FastAPI's interactive docs. Off in open mode: an open demo's API surface is
# the page, not an invitation to script it.
DOCS_PATHS = frozenset({"/docs", "/docs/oauth2-redirect", "/redoc", "/openapi.json"})


def _log_configuration() -> None:
    """
    One line at startup saying what this process will and will not do, so a
    public deploy that is missing a cap is visible in the first screen of logs.
    """
    fallback = judge_fallback_model()
    logger.info(
        "Vision %s; judge %s (fallback: %s); run timeout %ss",
        vision_model(),
        judge_model(),
        fallback or "none",
        int(job_timeout_seconds()) or "off",
    )
    mode = access_mode()
    logger.info(
        "Demo: access %s%s; %s calls/day (resets at midnight %s, %s%% kept for newcomers); "
        "%s calls/IP/hour; %s runs at once; clip <= %s s; up to %s lifts per clip; "
        "trusted proxies: %s; sample clip: %s%s",
        mode,
        "" if mode == "open" else (" (code set)" if access_required() else " (NO CODE SET)"),
        daily_limit() or "unlimited",
        getattr(day_timezone(), "key", "UTC"),
        int(_reserve_percent()),
        ip_hourly_limit() or "unlimited",
        max_concurrent_jobs() or "unlimited",
        int(max_clip_seconds()) or "any",
        max_lifts_per_clip() or "any",
        ", ".join(trusted_proxies()) or "none (X-Forwarded-For ignored)",
        "present" if load_sample() is not None else f"none at {sample_dir() or 'DEMO_SAMPLE_DIR (unset)'}",
        "; DEMO_DISABLED: scoring is OFF, the sample only" if demo_disabled() else "",
    )
    if demo_caps_set() and mode != "open" and not access_required():
        logger.warning(
            "Demo caps are set, ACCESS_MODE is %s and DEMO_ACCESS_CODE is not set: scoring is open to anyone who "
            "finds the URL, without open mode's startup check. Set a code, or set ACCESS_MODE=open on purpose.",
            mode,
        )
    if (demo_caps_set() or mode == "open") and load_sample() is None:
        logger.warning(
            "No sample clip is configured: once the budget is spent, visitors see nothing. "
            "Build one with tools/make_demo_sample.py and set DEMO_SAMPLE_DIR."
        )
    if ip_hourly_limit() and not trusted_proxies():
        logger.warning(
            "DEMO_IP_HOURLY_LIMIT is set but TRUSTED_PROXIES is not: behind a reverse proxy every "
            "visitor shares the proxy's address and its hourly allowance."
        )
    log_binaries()
    if ffmpeg_exe() is None:
        logger.warning(
            "No ffmpeg binary found: annotated replays will be skipped. "
            "Install `imageio-ffmpeg` (requirements.txt) or the apt buildpack with Aptfile=ffmpeg."
        )
    if ffprobe_exe() is None:
        logger.info(
            "ffprobe not on PATH: using OpenCV for video metadata. "
            "Portrait phone clips that declare rotation only in container metadata may be judged "
            "sideways; add the apt buildpack with Aptfile=ffmpeg for full rotation handling."
        )


def _reserve_percent() -> float:
    from server.services.demo_guard import daily_reserve_percent

    return daily_reserve_percent()


def _refuse_to_start_uncapped() -> None:
    """
    Open mode with a model key and a missing cap is a deploy that forgot one.
    Failing here, with the list, beats running an uncapped demo on a prepaid
    account; without a key nothing can be spent, so a local run still starts.
    """
    if access_mode() != "open" or not os.getenv("NEBIUS_API_KEY"):
        return
    missing = missing_open_mode_caps()
    if not missing:
        return
    message = (
        "ACCESS_MODE=open needs every spend cap set, and these are unset or zero: "
        + "; ".join(missing)
        + ". Set them (the deploy scripts do), or set ACCESS_MODE=code with a DEMO_ACCESS_CODE."
    )
    logger.error(message)
    raise RuntimeError(message)


@asynccontextmanager
async def lifespan(_: FastAPI):
    _refuse_to_start_uncapped()
    _log_configuration()
    yield
    # Never leave uploaded clips or replays behind (spec §12).
    get_store().shutdown()


def _refusal_response(exc: DemoRefused) -> JSONResponse:
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


def preflight(request: Request) -> None:
    """
    Everything that can turn a scoring request away from its headers alone:
    the kill switch, the access code in link and code modes, and a slot and
    budget for one lift from this address. Nothing is counted here.
    """
    check_enabled()
    check_access(request.headers.get("x-access-code"))
    get_guard().precheck(client_ip(request))


class DemoGateMiddleware:
    """
    Runs the demo's gate on the scoring routes before the request body is read.

    FastAPI parses the multipart form before any route code or dependency
    runs, so a check inside the handler happens only after the whole clip —
    up to MAX_UPLOAD_MB of it — has been received and spooled. This runs ahead
    of that: a paused demo, a missing or wrong code, a busy server or a spent
    budget is answered from the headers alone, and the body is never read.
    The code therefore travels as the X-Access-Code header, which the page
    sends and a script can too; a form field would be inside the body this
    is here not to read. In open mode the interactive API docs are off too.
    """

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] == "http":
            path = scope.get("path") or ""
            if path in DOCS_PATHS and access_mode() == "open":
                await JSONResponse(status_code=404, content={"detail": "Not Found"})(scope, receive, send)
                return
            if scope.get("method") == "POST" and path in GATED_PATHS:
                try:
                    preflight(Request(scope))
                except DemoRefused as exc:
                    await _refusal_response(exc)(scope, receive, send)
                    return
        await self.app(scope, receive, send)


app = FastAPI(title="Sideline Lift Judge", lifespan=lifespan)
app.add_middleware(DemoGateMiddleware)

# Mount static and templates
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

LIFT_CHOICES = [lift.value for lift in LiftType]


class ConfigurationError(RuntimeError):
    """Server-side misconfiguration; surfaced as a plain message, never a stack trace."""


@app.exception_handler(DemoRefused)
async def demo_refused(_: Request, exc: DemoRefused) -> JSONResponse:
    return _refusal_response(exc)


def _require_api_key() -> str:
    api_key = os.getenv("NEBIUS_API_KEY")
    if not api_key:
        raise ConfigurationError("The scoring service is not configured yet. Set NEBIUS_API_KEY on the server.")
    return api_key


def get_client(*, meter: Optional[Callable[[], None]] = None) -> NebiusChatClient:
    """
    The Token Factory client for one request. `meter` is charged before every
    model call, so the demo budget counts what is actually billed.
    """
    api_key = _require_api_key()
    base_url = os.getenv("NEBIUS_BASE_URL") or NebiusChatClient.DEFAULT_BASE_URL
    return NebiusChatClient(api_key=api_key, base_url=base_url, meter=meter)


T = TypeVar("T")


async def _within_job_timeout(work: Awaitable[T]) -> T:
    """
    Hold a whole scoring run to JOB_TIMEOUT_SECONDS. Past it the run is
    cancelled — no further model call is made — and the user hears that it was
    stopped, not that the server is busy. 0 disables the limit.
    """
    timeout = job_timeout_seconds()
    if not timeout:
        return await work
    return await asyncio.wait_for(work, timeout=timeout)


def _log_run(admission: Admission, *, status: str, kind: str = "job") -> None:
    """
    One structured line per admitted run: a keyed hash of the address (never
    the address), the calls it cost, what is left of the day, how long it
    took. No clip names: nothing in the log identifies a person or a file.
    """
    guard = get_guard()
    remaining = guard.remaining_today()
    logger.info(
        "%s status=%s ip=%s first_today=%s calls=%d remaining_today=%s running=%d duration=%.1fs",
        kind,
        status,
        ip_hash(admission.ip),
        "yes" if admission.first_today else "no",
        admission.calls,
        "unlimited" if remaining is None else remaining,
        guard.running,
        time.time() - admission.started_at,
    )


def _public_config() -> Dict[str, Any]:
    """What the page and a curious judge may know about this deploy. No secrets."""
    guard = get_guard()
    limit = daily_limit()
    return {
        "access_mode": access_mode(),
        "access_required": access_required(),
        "demo_disabled": demo_disabled(),
        "sample_available": load_sample() is not None,
        "max_clip_seconds": max_clip_seconds(),
        "max_upload_mb": max_upload_bytes() // (1024 * 1024),
        "max_lifts_per_clip": max_lifts_per_clip(),
        "daily_limit": limit or None,
        "daily_calls_remaining": guard.remaining_today() if limit else None,
        "vision": model_display_name(vision_model()),
        "judge": judge_display_name(),
    }


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    # Link-mode judges open the page with `?code=…` on the URL. Stripping it
    # client-side with history.replaceState leaves it visible until the JS
    # runs — long enough for a screenshot to leak it. Hand the code to a
    # short cookie the page reads once and redirect to a clean URL instead,
    # so the address bar never shows the secret.
    code = (request.query_params.get("code") or "").strip()
    if code:
        response = RedirectResponse(url=request.url.path, status_code=303)
        response.set_cookie(
            key="lj_access_code",
            value=code,
            max_age=120,
            path="/",
            secure=request.url.scheme == "https",
            httponly=False,
            samesite="lax",
        )
        return response
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
            "max_lifts_per_clip": max_lifts_per_clip(),
            "access_mode": access_mode(),
            "access_required": access_required(),
            "demo_disabled": demo_disabled(),
            "sample_available": load_sample() is not None,
            "judge_name": judge_display_name(),
            "vision_name": model_display_name(vision_model()),
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


@app.get("/api/config")
async def read_config():
    """The access mode and the caps in force, for the page and for whoever is checking the deploy."""
    return JSONResponse(content=_public_config())


@app.get("/api/access/check")
async def access_check(request: Request, code: Optional[str] = None):
    """
    Is this visitor allowed to score? Answered from the X-Access-Code header
    (or ?code=) with no model call, so the page can validate the code on
    "Next" — before anyone films or uploads — rather than after. Always yes
    in open mode; a paused demo says so here too.
    """
    check_enabled()
    check_access(request.headers.get("x-access-code") or code)
    return {"ok": True, "mode": access_mode()}


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
):
    """
    Accept a clip and start scoring in the background (spec §13). Returns the job
    id to poll, so a slow or dropped mobile connection cannot lose the result.
    """
    # Already checked before the body was read (DemoGateMiddleware); kept so
    # the handler is safe on its own.
    check_enabled()
    check_access(request.headers.get("x-access-code"))
    upload = _resolve_upload(video, file)
    selected_lift = _parse_lift(lift)
    start = _parse_optional_float(trim_start, "Trim start")
    end = _parse_optional_float(trim_end, "Trim end")
    mode = "auto" if str(mode).strip().lower() == "auto" else "single"

    # Fail before touching the disk if the service is not configured.
    try:
        _require_api_key()
    except ConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    guard = get_guard()
    ip = client_ip(request)
    store = get_store()
    job = store.create()
    admission: Optional[Admission] = None
    try:
        source = await _store_upload(upload, job.workdir)
        job.source_path = source
        _check_window(source, start, end)
        # Everything free has passed; only now does the run take a slot, so a
        # rejected window never costs the caller anything.
        admission = guard.admit(ip)
        client = get_client(meter=admission.meter())
    except (HTTPException, DemoRefused):
        if admission is not None:
            admission.release()
        store.discard(job)
        raise
    except ConfigurationError as exc:
        if admission is not None:
            admission.release()
        store.discard(job)
        raise HTTPException(status_code=503, detail=str(exc))

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
        max_lifts=max_lifts_per_clip(),
    )

    async def runner(current: Job) -> None:
        status = "done"
        try:
            outcome = await _within_job_timeout(
                run_scoring(
                    request_model,
                    client=client,
                    prompts_dir=str(PROMPTS_DIR),
                    progress=current.set_progress,
                    reserve=admission.reserve,
                )
            )
            current.replays = {
                item.index: Path(item.replay_path) for item in outcome.lifts if item.replay_path
            }
            current.stills = {
                f"{item.index}_{cue_id}": path
                for item in outcome.lifts
                for cue_id, path in item.still_paths.items()
                if path and Path(path).is_file()
            }
            _job_id = current.id
            current.payload = outcome_payload(
                outcome,
                replay_url=lambda index: f"/api/jobs/{_job_id}/replay/{index}.mp4",
                still_url=lambda index, cue_id: f"/api/jobs/{_job_id}/evidence/{index}_{cue_id}.jpg",
            )
            if not outcome.scored:
                first_error = next((item.error for item in outcome.lifts if item.error), None)
                if first_error:
                    raise UserFacingError(first_error)
                current.set_progress("done", "No snatch, clean or jerk found in this clip.", 1.0)
                return
            current.set_progress("done", "Done.", 1.0)
        except BaseException:
            status = "error"
            raise
        finally:
            admission.release()
            _log_run(admission, status=status)

    store.start(job, runner)
    return JSONResponse(status_code=202, content={"id": job.id, "poll_url": f"/api/jobs/{job.id}"})


@app.get("/api/jobs/{job_id}")
async def read_job(job_id: str):
    job = get_store().get(job_id)
    if job is None:
        # The in-memory store loses jobs both when the TTL elapses and when
        # the dyno restarts (`heroku config:set` cycles it). The second case
        # is by far the more common one for a user polling a few seconds
        # after submitting, so lead with it.
        raise HTTPException(
            status_code=404,
            detail="The server restarted while this was running. Please try again.",
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


@app.get("/api/jobs/{job_id}/evidence/{key}.jpg")
async def read_evidence_still(job_id: str, key: str):
    job = get_store().get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="That result has expired.")
    path = job.stills.get(key)
    if path is None or not Path(path).is_file():
        raise HTTPException(status_code=404, detail="No evidence still for that cue.")
    return FileResponse(
        path,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=1800"},
    )


@app.get("/api/sample")
async def read_sample():
    """
    The cached, pre-scored sample clip: free to serve, no access code, not
    counted against any limit, and still there when the demo is paused. What
    a judge sees when the budget is spent.
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
):
    """
    Synchronous scoring: one request, one verdict. Kept for scripts and for
    parity with the CLI; the web UI uses /api/jobs so it can show progress.
    """
    check_enabled()
    check_access(request.headers.get("x-access-code"))
    upload = _resolve_upload(video, file)
    selected_lift = _parse_lift(lift)
    start = _parse_optional_float(trim_start, "Trim start")
    end = _parse_optional_float(trim_end, "Trim end")

    source = await _store_upload(upload)
    guard = get_guard()
    ip = client_ip(request)
    admission: Optional[Admission] = None
    status = "error"
    try:
        _require_api_key()
        _check_window(source, start, end)
        admission = guard.admit(ip)
        client = get_client(meter=admission.meter())
        outcome = await _within_job_timeout(
            run_scoring(
                ScoreRequest(
                    video_path=str(source),
                    lift=selected_lift,
                    trim_start=start,
                    trim_end=end,
                    frames=clamp_frames(frames if frames is not None else num_frames),
                    mode="auto" if str(mode).strip().lower() == "auto" else "single",
                    render_replay=False,
                    keep_observations=bool(debug),
                    max_lifts=max_lifts_per_clip(),
                ),
                client=client,
                prompts_dir=str(PROMPTS_DIR),
                reserve=admission.reserve,
            )
        )
        scored = outcome.scored
        if not scored:
            first_error = next((item.error for item in outcome.lifts if item.error), None)
            raise HTTPException(
                status_code=422,
                detail=first_error or "No snatch, clean or jerk could be scored in this clip.",
            )
        status = "done"
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
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail=TOO_LONG)
    except Exception as exc:
        logger.exception("Scoring failed")
        raise HTTPException(
            status_code=502,
            detail="Scoring failed. The model service may be busy — wait a moment and try the clip again.",
        ) from exc
    finally:
        if admission is not None:
            admission.release()
            _log_run(admission, status=status, kind="score")
        Path(source).unlink(missing_ok=True)


@app.get("/healthz")
async def healthz():
    return {"ok": True}
