#!/usr/bin/env bash
# Deploy the public demo to Heroku as one container, on one dyno.
#
#   heroku login                                        # once; opens the browser
#   APP=lift-judge-demo DEMO_ACCESS_CODE=the-code deploy/heroku.sh      # gated (code mode)
#   APP=lift-judge-demo ACCESS_MODE=open deploy/heroku.sh               # open, caps on
#
# The image is built locally and pushed to Heroku's registry, so the cached
# sample in demo_sample/ ships inside it whether or not it is committed.
# Re-running the script redeploys. Optional: REGION (default eu, nearest
# Token Factory), DYNO_TYPE (e.g. basic: the cheapest dyno that does not
# sleep), ACCESS_MODE (open | link | code; unset keeps the app's current
# setting, or `code` on a new app), MAX_CONCURRENT_JOBS (default 1 — a 512 MB
# dyno), MAX_UPLOAD_MB (default 60: Heroku's router gives a request 30 s,
# upload included).
#
# The spend caps are the image's defaults (see the Dockerfile); open mode
# refuses to start without them. Heroku's router is the only thing that
# reaches the dyno and it appends the visitor's address as the last
# X-Forwarded-For entry, which is what TRUSTED_PROXIES='*' reads.
set -euo pipefail

cd "$(dirname "$0")/.."

: "${APP:?set APP to the Heroku app name}"
REGION="${REGION:-eu}"
ACCESS_MODE="${ACCESS_MODE:-}"
if [ "$ACCESS_MODE" != "open" ] && [ -z "${DEMO_ACCESS_CODE:-}" ]; then
  echo "set DEMO_ACCESS_CODE (link and code modes need one), or ACCESS_MODE=open to run the demo with no code" >&2
  exit 1
fi

# The key: from the environment, else from .env. Use a key made just for the
# demo, so it can be revoked without touching the one you record with.
if [ -z "${NEBIUS_API_KEY:-}" ] && [ -f .env ]; then
  NEBIUS_API_KEY="$(grep -E '^NEBIUS_API_KEY=' .env | tail -1 | cut -d= -f2- | tr -d '"'"'"' ')"
fi
: "${NEBIUS_API_KEY:?set NEBIUS_API_KEY, or put it in .env}"

if [ ! -f demo_sample/sample.json ]; then
  echo "note: demo_sample/ holds no sample. Build one first so the demo has a free fallback:" >&2
  echo "      python tools/make_demo_sample.py samples/<clip>.mp4 --lift snatch --out demo_sample/" >&2
  if [ "$ACCESS_MODE" = "open" ]; then
    echo "      An open demo without it is a dead end once the budget is spent; stopping." >&2
    exit 1
  fi
fi

if heroku apps:info -a "$APP" >/dev/null 2>&1; then
  heroku stack:set container -a "$APP" >/dev/null
else
  heroku apps:create "$APP" --stack container --region "$REGION"
fi

heroku config:set -a "$APP" \
  NEBIUS_API_KEY="$NEBIUS_API_KEY" \
  TRUSTED_PROXIES='*' \
  MAX_CONCURRENT_JOBS="${MAX_CONCURRENT_JOBS:-1}" \
  MAX_UPLOAD_MB="${MAX_UPLOAD_MB:-60}"
if [ -n "${DEMO_ACCESS_CODE:-}" ]; then
  heroku config:set -a "$APP" DEMO_ACCESS_CODE="$DEMO_ACCESS_CODE"
fi
if [ -n "$ACCESS_MODE" ]; then
  heroku config:set -a "$APP" ACCESS_MODE="$ACCESS_MODE"
fi

heroku container:login
heroku container:push web -a "$APP"
heroku container:release web -a "$APP"
heroku ps:scale web=1 -a "$APP"
if [ -n "${DYNO_TYPE:-}" ]; then
  heroku ps:type "$DYNO_TYPE" -a "$APP"
fi

URL="$(heroku apps:info -a "$APP" -s | sed -n 's/^web_url=//p')"
echo
echo "Deployed: ${URL}"
if [ "$ACCESS_MODE" = "open" ]; then
  echo "Judges open: ${URL} (no code)"
else
  echo "Judges open: ${URL}?code=<the access code>"
fi
echo "Check: curl -s ${URL}api/config ; curl -sI ${URL}api/sample | head -1"
echo "First lines of the log (the caps in force):"
sleep 15
heroku logs -a "$APP" --num 40 2>/dev/null | grep -E "Vision |Demo:|WARNING|ERROR" || echo "(not up yet — run: heroku logs --tail -a $APP)"
