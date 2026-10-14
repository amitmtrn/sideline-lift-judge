import base64
import os
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv

from app.schemas.judgment import LiftJudgment, LiftType
from app.utils.frames import sample_video_frames_bgr
from app.utils.video_meta import get_rotation_degrees
from app.utils.image_ops import rotate_by_degrees, resize_long_side
from app.services.nebius_client import (
    NO_THINKING_EXTRA_BODY,
    NebiusChatClient,
    assemble_vision_messages,
    request_judgment,
)


# Load .env if present for local dev
load_dotenv()

APP_DIR = Path(__file__).resolve().parent
BASE_DIR = APP_DIR.parent
PROMPTS_DIR = BASE_DIR / "prompts"
TEMPLATES_DIR = APP_DIR / "templates"
STATIC_DIR = APP_DIR / "static"

app = FastAPI(title="Sideline Lift Judge — MVP")

# Mount static and templates
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


def bgr_frames_to_data_urls(frames_bgr: List["np.ndarray"]) -> List[str]:
    # Lazy import to avoid cv2 dependency on import path for test discovery
    import cv2  # type: ignore
    data_urls: List[str] = []
    for frame in frames_bgr:
        success, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not success:
            continue
        b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
        data_urls.append(f"data:image/jpeg;base64,{b64}")
    return data_urls


def get_client() -> NebiusChatClient:
    api_key = os.getenv("NEBIUS_API_KEY")
    if not api_key:
        raise RuntimeError("NEBIUS_API_KEY is not set. See README for setup.")
    base_url = os.getenv("NEBIUS_BASE_URL") or NebiusChatClient.DEFAULT_BASE_URL
    return NebiusChatClient(api_key=api_key, base_url=base_url)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "default_lift": "snatch",
        },
    )


@app.post("/api/score")
async def score_endpoint(
    request: Request,
    lift: LiftType = Form(...),
    file: UploadFile = File(...),
    num_frames: int = Form(10),
):
    if file.content_type not in ("video/mp4", "video/quicktime", "video/x-matroska", "video/mpeg", "video/avi"):
        raise HTTPException(status_code=400, detail=f"Unsupported content type: {file.content_type}")
    # Persist upload to temp file for OpenCV
    import tempfile

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename or "clip").suffix) as tmp:
        data = await file.read()
        tmp.write(data)
        tmp_path = Path(tmp.name)

    try:
        frames = sample_video_frames_bgr(str(tmp_path), target_count=max(8, min(16, int(num_frames))))
        if not frames:
            raise HTTPException(status_code=400, detail="Could not decode frames from the uploaded video.")
        # Orientation + downscale to ~768 long side
        rotation_deg = get_rotation_degrees(str(tmp_path))
        processed = [resize_long_side(rotate_by_degrees(f, rotation_deg), 768) for f in frames]
        images = bgr_frames_to_data_urls(processed)
        client = get_client()
        # Stage 1: vision observations (MiniCPM-V) with chunking when needed
        vision_model = os.getenv("NEBIUS_VISION_MODEL") or "openbmb/MiniCPM-V-4_5"
        max_images = int(os.getenv("NEBIUS_VISION_MAX_IMAGES") or "10")
        if len(images) <= max_images:
            v_messages = assemble_vision_messages(lift=lift, images=images, prompts_dir=str(PROMPTS_DIR))
            observations = await client.create_chat_completion(
                messages=v_messages,
                model=vision_model,
                max_tokens=1500,
                extra_body=NO_THINKING_EXTRA_BODY,
            )
        else:
            from app.services.nebius_client import collect_observations_in_chunks
            observations = await collect_observations_in_chunks(
                client=client,
                lift=lift,
                images=images,
                prompts_dir=str(PROMPTS_DIR),
                model=vision_model,
                max_images_per_call=max_images,
            )
        # Stage 2: judge (Nemotron)
        judge_model = os.getenv("NEBIUS_JUDGE_MODEL") or "nvidia/Nemotron-3-Ultra-550b-a55b"
        # Call with one automatic retry on parse failure
        judgment = await request_judgment(
            client=client,
            lift=lift,
            observations=observations,
            prompts_dir=str(PROMPTS_DIR),
            model=judge_model,
        )
        return JSONResponse(content=judgment.model_dump())
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@app.get("/healthz")
async def healthz():
    return {"ok": True}

