#!/usr/bin/env bash
# The web process for a Heroku *buildpack* deploy (see the Procfile): what a
# dashboard "Deploy Branch" or a `git push heroku` on the default stack runs.
#
# The Docker image carries the public-demo defaults as ENV; this gives the
# buildpack build the very same ones, so the demo is capped the same way
# however it was deployed — and so ACCESS_MODE=open, which refuses to start
# without every one of them, starts. Each default applies only when the
# variable is unset, so a config var set in the dashboard or with
# `heroku config:set` wins; ACCESS_MODE itself has no default here (the code
# treats unset as `code`, gated) and is the one var the rollout flips:
#
#   heroku config:set ACCESS_MODE=open  -a <app>     # no code, caps on
#   heroku config:set ACCESS_MODE=link  -a <app>     # back to the ?code= link
#   heroku config:set DEMO_DISABLED=true -a <app>    # the sample only
#
# Heroku-specific: its router is the only thing that reaches the dyno and
# appends the visitor as the last X-Forwarded-For entry (TRUSTED_PROXIES '*'),
# a 512 MB dyno fits one run at a time, and the router gives a request 30 s
# including the upload (a 60 MB cap makes the page warn from 30 MB). The day's
# budget turns at midnight in Israel, where the judging is.
set -euo pipefail

export NEBIUS_JUDGE_MODEL="${NEBIUS_JUDGE_MODEL:-nvidia/nemotron-3-super-120b-a12b}"
# `-` not `:-`: an unset fallback becomes empty (disabled), an explicit one stays.
export NEBIUS_JUDGE_FALLBACK_MODEL="${NEBIUS_JUDGE_FALLBACK_MODEL-}"
export DEMO_DAILY_LIMIT="${DEMO_DAILY_LIMIT:-150}"
export DEMO_DAY_TIMEZONE="${DEMO_DAY_TIMEZONE:-Asia/Jerusalem}"
export DEMO_DAILY_RESERVE_PERCENT="${DEMO_DAILY_RESERVE_PERCENT:-20}"
export DEMO_IP_HOURLY_LIMIT="${DEMO_IP_HOURLY_LIMIT:-12}"
export MAX_CONCURRENT_JOBS="${MAX_CONCURRENT_JOBS:-1}"
export MAX_CLIP_SECONDS="${MAX_CLIP_SECONDS:-15}"
export MAX_LIFTS_PER_CLIP="${MAX_LIFTS_PER_CLIP:-2}"
export MAX_FRAMES="${MAX_FRAMES:-10}"
export MAX_UPLOAD_MB="${MAX_UPLOAD_MB:-60}"
export JOB_TIMEOUT_SECONDS="${JOB_TIMEOUT_SECONDS:-120}"
export TRUSTED_PROXIES="${TRUSTED_PROXIES:-*}"
export DEMO_SAMPLE_DIR="${DEMO_SAMPLE_DIR:-demo_sample}"

exec uvicorn server.main:app --host 0.0.0.0 --port "${PORT:-8000}" --workers 1 --timeout-keep-alive 75
