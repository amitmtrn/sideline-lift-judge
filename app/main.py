import base64
import os
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError
from dotenv import load_dotenv

from app.schemas.judgment import LiftJudgment, LiftType, parse_and_validate_json
from app.utils.frames import sample_video_frames_bgr
from app.services.nebius_client import NebiusChatClient, assemble_messages_for_lift


# Load .env if present for local dev
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
PROMPTS_DIR = BASE_DIR / "prompts"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

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
    model = os.getenv("NEBIUS_MODEL") or NebiusChatClient.DEFAULT_MODEL
    return NebiusChatClient(api_key=api_key, base_url=base_url, default_model=model)


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
    num_frames: int = Form(12),
):
    if file.content_type not in ("video/mp4", "video/quicktime", "video/x-matroska", "video/mpeg", "video/avi"):
        raise HTTPException(status_code=400, detail=f"Unsupported content type: {file.content_type}")
    # Persist upload to temp file for OpenCV
    import tempfile

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename or \"clip\").suffix) as tmp:
        data = await file.read()
        tmp.write(data)
        tmp_path = Path(tmp.name)

    try:
        frames = sample_video_frames_bgr(str(tmp_path), target_count=max(8, min(16, int(num_frames))))
        if not frames:
            raise HTTPException(status_code=400, detail="Could not decode frames from the uploaded video.")
        images = bgr_frames_to_data_urls(frames)
        client = get_client()
        messages = assemble_messages_for_lift(lift=lift, images=images, prompts_dir=str(PROMPTS_DIR))
        # Call with one automatic retry on parse failure
        first = await client.create_chat_completion(messages=messages, model=None)
        try:
            judgment = parse_and_validate_json(first)
        except (ValueError, ValidationError):
            # Add a terse repair instruction and retry once
            retry_messages = messages + [
                {
                    "role": "system",
                    "content": "Your previous response was not valid JSON. Reply again with JSON only matching the schema, no markdown.",
                }
            ]
            second = await client.create_chat_completion(messages=retry_messages, model=None)
            judgment = parse_and_validate_json(second)  # Let errors bubble
        return JSONResponse(content=judgment.model_dump())
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@app.get("/healthz")
async def healthz():
    return {"ok": True}

