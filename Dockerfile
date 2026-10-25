# Sideline Lift Judge — one container, one process (jobs are held in memory).
#
#   python tools/make_demo_sample.py samples/snatch1.mp4 --lift snatch --out demo_sample/
#   docker build -t lift-judge .
#   docker run --rm -p 8000:8000 --env-file .env lift-judge
#
# ffmpeg/ffprobe do the trimming, rotation metadata and the annotated replay;
# the DejaVu font is what the replay's verdict strip is drawn with. Camera
# capture and PWA install need HTTPS in front of this (see deploy/).
#
# This image is the public demo, so the spend caps are ON here, not opt-in.
# A container's own environment (--env-file, -e, a platform's settings)
# overrides any ENV below; the local checkout keeps the code defaults (Ultra
# with a Super fallback, no caps), which is what the recorded run uses.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

# Public-demo defaults: every cap ACCESS_MODE=open insists on, at the values
# the PRD sets. ACCESS_MODE itself is not set here (so it is `code`, gated):
# the deploy flips it to `open` with one config var once the caps and the
# sample are verified, and back to `link` or `code` — or DEMO_DISABLED=true —
# without a rebuild if the spend looks wrong. DEMO_ACCESS_CODE has no default
# (it is the secret) and is only needed in link and code modes. The judge is
# Nemotron 3 Super with no fallback to Ultra; the budgets count model calls
# (a single lift is three) and the day turns at midnight in Israel;
# TRUSTED_PROXIES stays unset so X-Forwarded-For is ignored until the deploy
# says which proxy to believe (deploy/ sets it).
ENV NEBIUS_JUDGE_MODEL=nvidia/nemotron-3-super-120b-a12b \
    NEBIUS_JUDGE_FALLBACK_MODEL= \
    DEMO_DAILY_LIMIT=150 \
    DEMO_DAY_TIMEZONE=Asia/Jerusalem \
    DEMO_DAILY_RESERVE_PERCENT=20 \
    DEMO_IP_HOURLY_LIMIT=12 \
    MAX_CONCURRENT_JOBS=1 \
    MAX_CLIP_SECONDS=15 \
    MAX_LIFTS_PER_CLIP=2 \
    MAX_FRAMES=10 \
    MAX_UPLOAD_MB=60 \
    JOB_TIMEOUT_SECONDS=120 \
    DEMO_SAMPLE_DIR=/app/demo_sample

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg fonts-dejavu-core libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY server server
COPY client client
COPY prompts prompts
COPY tools tools
COPY score_clip.py score_session.py ./
# The cached sample clip, if one was built (tools/make_demo_sample.py): the
# free result a visitor sees once the budget is spent. The directory is
# tracked (its .gitkeep at least) so this COPY always has something to copy.
COPY demo_sample demo_sample

# Uploads, replays and the job store live under /tmp; nothing is persisted.
RUN useradd --create-home --uid 10001 app && chown -R app /app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request,os; urllib.request.urlopen('http://127.0.0.1:%s/healthz' % os.environ.get('PORT','8000'))" || exit 1

# One worker on purpose: jobs and demo counters are in-process. The visitor's
# address is worked out by the app from TRUSTED_PROXIES, so uvicorn is not
# told to believe X-Forwarded-For from everyone.
CMD ["sh", "-c", "uvicorn server.main:app --host 0.0.0.0 --port ${PORT} --workers 1 --timeout-keep-alive 75"]
