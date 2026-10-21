# Sideline Lift Judge — one container, one process (jobs are held in memory).
#
#   docker build -t lift-judge .
#   docker run --rm -p 8000:8000 --env-file .env lift-judge
#
# ffmpeg/ffprobe do the trimming, rotation metadata and the annotated replay;
# the DejaVu font is what the replay's verdict strip is drawn with. Camera
# capture and PWA install need HTTPS in front of this (see deploy/).
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

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

# Uploads, replays and the job store live under /tmp; nothing is persisted.
RUN useradd --create-home --uid 10001 app && chown -R app /app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request,os; urllib.request.urlopen('http://127.0.0.1:%s/healthz' % os.environ.get('PORT','8000'))" || exit 1

# One worker on purpose: jobs and demo counters are in-process.
CMD ["sh", "-c", "uvicorn server.main:app --host 0.0.0.0 --port ${PORT} --workers 1 --timeout-keep-alive 75 --proxy-headers --forwarded-allow-ips='*'"]
