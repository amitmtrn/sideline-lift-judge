# Sideline Lift Judge — Product Spec

> **Status:** Hackathon MVP spec · **Owner:** Amit Avigad · **Last updated:** Oct 17, 2026
> **Repo:** https://github.com/amitmtrn/sideline-lift-judge (public, MIT license — PR #2)
> **Event:** Nebius × NVIDIA Global AI Hackathon
> **Deadline:** Fri **Oct 30, 2026, 10:00 PDT** = **19:00 Israel time (IST, UTC+2)**. Israel leaves summer time on Oct 25, so it's 19:00, not 20:00. Plan to submit on Oct 28.
> **Language:** English only. No Hebrew anywhere in the UI, prompts or output.

---

## 1. One-liner

Upload or film a short clip of one Olympic lift. You get back a **1–10 bar-path efficiency score**, a **band**, a description of the **maximum horizontal bar drift**, a **confidence** value, **notes**, and **1–2 coaching cues**. The app is mobile-first, so a coach can use it from the side of the platform.

## 2. Problem & audience

- **Who it's for:** weightlifting coaches and athletes, mostly amateur or club level, who film lifts on a phone and want quick, objective feedback on bar path.
- **The problem:** You can't see bar path well in real time. Frame-by-frame review takes time and needs expertise. Most feedback comes down to one principle: *keep the bar close.*
- **What we offer:** One clip goes in. A structured, rubric-based verdict and one or two actionable cues come out in under a minute, right on the phone.

## 3. Scope

### In scope (by priority)
| Priority | Lift | Notes |
|---|---|---|
| P0 (demo) | **Snatch** | The main demo lift |
| P1 | **Clean** | Second lift |
| P2 (stretch) | **Jerk** | A clean & jerk is scored as **two separate calls**: clean, then jerk |

### Out of scope
- Squat, deadlift and other non-Olympic lifts
- A chat or conversational coach
- Hebrew or any other non-English UI
- Accounts, logins, payments
- Pose estimation, custom ML training or fine-tuning
- Native iOS/Android apps for the hackathon (post-hackathon stretch only; see §9)

## 4. Hackathon rules & track decision

### Hard requirements
- [ ] Runs on **Nebius Token Factory** or **Nebius AI Cloud**
- [ ] Uses **at least one NVIDIA open model** (Nemotron 3 Ultra is the judge)
- [ ] Public repo with a visible OSS license (MIT ✅)
- [ ] README
- [ ] YouTube video, **≤ 3 minutes**
- [ ] Demo URL (not required for the Physical AI track, but we'll ship one anyway)
- [ ] Tool feedback form

**Judging criteria:** technical implementation, design, impact, idea.

### Track decision

| | **Physical AI** | **Best Apps and Agents** |
|---|---|---|
| Listed models | Cosmos / GR00T / Nemotron | Nemotron on Token Factory |
| How we fit | The subject is physical: human motion and a barbell | Strong fit: an app built on Token Factory with a Nemotron judge |
| Blocker | **Amit's Token Factory account has no Cosmos models.** The NVIDIA models we can use take no image input, so the visual stage runs on a non-NVIDIA model (MiniCPM-V) | None found |
| Demo URL | Optional | Required (we have one: the PWA) |

**Recommendation: Best Apps and Agents.** It matches exactly what runs today: Nemotron on Token Factory, a working deployed app, and a mobile-first demo URL. Physical AI judges would likely expect Cosmos or GR00T for the visual and physical reasoning, which we can't provide.

**Risk:** organizers may see the vision stage as "non-NVIDIA". **Mitigation:** the README and video state plainly that the NVIDIA open model (Nemotron 3 Ultra) does the reasoning and scoring, and Token Factory hosts both stages. If Cosmos shows up on the account before Oct 24, re-check the Physical AI track. Confirm the track wording with the organizers before submitting.

## 5. Verified technical reality (as of Oct 14)

- **No Cosmos models** on Amit's Token Factory account.
- These NVIDIA models **reject image input**: `nvidia/Nemotron-3-Ultra-550b-a55b`, `nvidia/nemotron-3-super-120b-a12b`, `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B`, `nvidia/Nemotron-3_5-Lightning`.
- So the pipeline has **two stages** (PR #3 merged; PR #4 merged with fixes):
  1. **Vision:** `openbmb/MiniCPM-V-4_5` describes the bar path for each phase, from sampled frames.
  2. **Judge:** `nvidia/Nemotron-3-Ultra-550b-a55b` applies the rubric to that description and returns validated JSON.

## 6. MVP definition

The MVP is done when someone can open a public demo URL **on a phone**, film or upload a **side-view snatch**, optionally trim it to one lift, and within about a minute see:
- a score from 1 to 10 and a band
- a description of max horizontal drift
- a confidence value that reflects clip quality and camera angle
- notes, plus 1–2 coaching cues
- an annotated replay of the clip that plays inline and can be downloaded or shared

All of this runs on Nebius Token Factory, with an NVIDIA Nemotron model as the judge, and needs no account and no desktop.

## 7. User stories

1. **Coach (P0):** As a coach at the gym, I film my athlete's snatch from the side on my phone and get a bar-path score and one cue before the next attempt.
2. **Athlete (P0):** As an athlete, I upload a clip from my gallery and learn whether the bar drifted away from me, and in which phase.
3. **Coach (P0):** As a coach, I trim a long clip to just the lift so the score covers one attempt.
4. **Any user (P0):** As a user, I see a confidence value and a warning when my clip is badly angled or mixes lifts, so I know how far to trust the score.
5. **Athlete (P1):** As an athlete, I score my clean separately from my jerk.
6. **Coach (P1):** As a coach, I download or share the annotated replay with my athlete over WhatsApp or the share sheet.
7. **Judge or evaluator (P0):** As a hackathon judge, I open the demo URL on my phone and get a result without installing anything.
8. **Developer (P0):** As a developer, I run `score_clip.py` on a local clip with `--debug` and inspect the output from both stages.

## 8. UX flow (mobile-first)

```
[Home / PWA icon]
   │
   ▼
1. Choose lift: Snatch (default) · Clean · Jerk
   │
   ▼
2. Get clip:  [🎥 Film now]  or  [🖼 Upload from gallery]
   │           └─ Filming guide overlay (side-view silhouette, level line,
   │              "full bar in frame" check, landscape hint)
   ▼
3. Trim: start/end sliders + preview (one lift per clip)
   │
   ▼
4. Upload & analyze: progress states
   "Compressing… → Uploading… → Reading bar path… → Judging… → Rendering replay…"
   │
   ▼
5. Result screen
   • Large score (e.g. 3/10) + band badge
   • Max horizontal drift (plain-language description)
   • Confidence meter + any angle/quality warnings
   • 1–2 cues as large cards
   • Notes (collapsible)
   • Annotated video plays inline → [Download MP4] [Share]
   │
   ▼
6. [Score another lift]  (e.g. score the jerk after the clean)
```

### UX principles
- **Readable from a distance:** large type for score and cues, high contrast, controls that work with chalky thumbs.
- **Honesty first:** confidence and warnings sit right next to the score, never hidden.
- **Filming guidance appears before the camera opens**, not after a bad result.
- **English only**, short coaching language.

### Filming guidance (shown in-app and in the README)
- Film from **square to the side**, **3–5 m away**, with the camera at **hip height**
- **Landscape preferred**; portrait works but lowers confidence
- **The whole bar and both plates stay in frame** for the entire lift
- **One lift per clip.** Trim a clean & jerk into two clips.

## 9. Mobile

### Recommendation: PWA for the MVP, native after the hackathon

**MVP: a mobile-first responsive web app, installable as a PWA.**
- Reuses the existing **FastAPI + Jinja** UI; no new front-end stack
- Installable: web app manifest, home-screen icon, standalone display
- Camera capture via `<input type="file" accept="video/*" capture="environment">`, with an optional `getUserMedia` / `MediaRecorder` path for the guide overlay
- Upload from the gallery or camera roll
- Targets **iOS Safari** and **Android Chrome**
- **No app-store review.** Judges get a demo URL right away, which fits the Oct 30 deadline.

**Post-hackathon stretch: native app (React Native or Flutter).**
- Gains: finer camera control (60 fps, locked exposure and focus, a live angle-guide overlay), on-device trimming, push notifications, an offline upload queue
- Why not now: store review, build and signing setup, and testing on two platforms don't fit the deadline

### Mobile-specific UX requirements
| Feature | MVP behavior |
|---|---|
| Filming guide overlay | Side-view silhouette, horizon/level line, "full bar in frame" frame box, landscape hint. On the `getUserMedia` path it's a live overlay; on the `<input capture>` path it's a guide screen shown before the native camera opens |
| Trim | Start/end sliders with touch-friendly handles and a preview loop; trim times go to the server (no re-encode on the phone) |
| Result | Large score and cue cards that fit one screen without scrolling |
| Annotated video | Plays inline (`playsinline`, muted autoplay allowed); **MP4 / H.264 + AAC (or no audio), yuv420p, faststart** so iOS can play it; Download button plus Web Share API (`navigator.share` with files) where available, otherwise a download link |
| Networks | Works over gym Wi-Fi or 4G: optional client-side downscale before upload, server samples frames at ~768 px, and clear progress states with retry |

### Mobile risks & mitigations
| Risk | Mitigation |
|---|---|
| iOS Safari HEVC/.mov uploads and **rotation metadata** quirks (already hit with portrait phone video) | Server normalizes rotation (existing handling); transcode or decode HEVC server-side with ffmpeg; test with real iPhone and Android clips |
| Large files / upload timeouts | Trim before upload; client-side size check and warning; server upload limit with a clear error; generous timeouts on the reverse proxy; show upload progress |
| Token Factory latency on mobile networks | Only frames go to the model (≤ 10 images/request, ~768 px), never the full video; staged progress UI; async job + polling so a dropped connection doesn't lose the result |
| Annotated-video re-encode time | Encode only the trimmed segment, at modest resolution and fast preset; show the score first and let the replay arrive after |
| Camera API differences | Treat `<input capture>` as the baseline that always works; `getUserMedia` overlay is progressive enhancement |
| PWA install differences (iOS has no install prompt) | "Add to Home Screen" hint on iOS; the app works fully without installing |

## 10. Input / output spec

### Input
| Field | Type | Rules |
|---|---|---|
| `video` | file | Common phone formats (MP4/H.264, MOV, HEVC). One lift. Must never be committed to the repo |
| `lift` | enum | `snatch` \| `clean` \| `jerk` |
| `trim_start`, `trim_end` | float seconds | Optional; must satisfy `0 ≤ start < end ≤ duration` |
| `frames` | int | Optional; clamped to **8–16**, default **10** |

### Output (validated Pydantic model)
| Field | Type | Meaning |
|---|---|---|
| `lift` | enum | The lift that was scored |
| `score` | int 1–10 | Bar-path efficiency score |
| `band` | enum | Qualitative band (e.g. `poor`); bands map to fixed score ranges in the rubric |
| `max_horizontal_drift` | string | Plain-language description of where and how far the bar drifted from the body |
| `cues` | string[1–2] | Short, actionable coaching cues |
| `confidence` | float 0–1 | How much to trust the score, based on clip quality, angle and phase clarity |
| `notes` | string | Caveats: angle, mixed lifts, occlusion, and so on |
| `annotated_video_url` | string (MVP addition) | Link to the iOS-safe MP4 replay |
| `warnings` | string[] (MVP addition) | Clip-quality warnings shown next to the score (FR-8) |

`score` and `confidence` are `null` and `band` is `unknown` when the media is unusable — the one case where `cues` may be empty.

### Example: first real run (clean & jerk clip, scored as `clean`)
```json
{
  "lift": "clean",
  "score": 3,
  "band": "poor",
  "max_horizontal_drift": "<as returned by the judge — not reproduced here>",
  "cues": [
    "keep the bar close to the body",
    "bar close into the hips; contact then explode up"
  ],
  "confidence": 0.85,
  "notes": "<as returned by the judge — not reproduced here>"
}
```
> ⚠️ **Known problem:** the clip mixed a clean and a jerk, and the vision stage blended the phases together ("catch overhead, legs split"). A **confidence of 0.85 was too high** for that input. The confidence rule needs fixing (see FR-8). Until it is, and until clips are trimmed to one lift and filmed from the side, **treat scores as indicative only**.

## 11. Functional requirements

| ID | Requirement | Priority |
|---|---|---|
| FR-1 | Accept a video upload or in-browser camera capture, plus a lift selection | P0 (done) |
| FR-2 | Trim the clip to start/end on the server before frame sampling | P0 (done) |
| FR-3 | Sample 8–16 frames (default 10), resize to ~768 px, and normalize rotation metadata | P0 (exists) |
| FR-4 | **Stage 1:** send frames to the vision model (≤ 10 images/request, chunk if more) and get a bar-path description for each phase; strip or disable `<think>` output | P0 (exists) |
| FR-5 | **Stage 2:** send the observation plus the lift-specific rubric prompt to the Nemotron judge; parse and validate against the Pydantic schema; retry or repair once on invalid JSON | P0 (exists) |
| FR-6 | Fall back to `nemotron-3-super-120b-a12b` if Ultra fails or times out | P0 (exists) |
| FR-7 | Return 1–2 cues tied to the observed fault and grounded in the biomechanics principles (§14) | P0 (done) |
| FR-8 | **Angle-aware confidence:** lower confidence for non-side views, portrait video, missing bar, multiple lifts or unclear phases; show a matching warning in `notes` and the UI | P0 (done — `server/services/confidence.py`) |
| FR-9 | Detect a likely mixed clean & jerk and prompt the user to trim or split it | P1 (done — it is split and both lifts scored) |
| FR-10 | Render an annotated replay MP4 (H.264, iOS-safe) with inline playback, download and share | P0 (done) |
| FR-11 | Mobile-first responsive UI + PWA manifest, icon and installability | P0 (done) |
| FR-12 | Filming guide overlay or screen before capture | P0 (done — guide screen before the camera opens) |
| FR-13 | CLI `score_clip.py --lift --frames --debug` stays in sync with the web pipeline | P0 (exists) |
| FR-14 | Support for Jerk (`user_jerk.txt`), and clean & jerk scored as two calls | P2 (done) |

## 12. Non-functional requirements

- **Platform:** inference only on Nebius Token Factory (`https://api.tokenfactory.nebius.com/v1`); at least one NVIDIA open model in the path.
- **Latency:** results should arrive in roughly a minute on a normal connection, with progress states the whole time. This is a target, not a measured figure.
- **Mobile:** works on current iOS Safari and Android Chrome; touch targets ≥ 44 px; no horizontal scroll.
- **Reliability:** schema-validated output, with a graceful error message and no stack traces shown to users.
- **Privacy & security:** uploaded videos are temporary and deleted after processing or after a short TTL. **Never commit videos, API keys or course PDFs.** Secrets come only from env vars, with `.env` gitignored.
- **Honesty:** confidence and caveats always appear next to the score; nothing claims accuracy beyond what we've validated.
- **Language:** all UI, prompts and outputs in English.
- **License:** MIT, visible in the repo.
- **Testability:** unit tests for frame sampling, rotation, chunking, `<think>` stripping and schema validation; mock the Token Factory calls in CI.

## 13. Architecture

```
 Phone (iOS Safari / Android Chrome, PWA)
   │  film / upload + lift + trim
   ▼
 FastAPI + Jinja (mobile-first templates, PWA manifest, service worker)
   │
   ├─► Video prep (ffmpeg / OpenCV)
   │     trim → rotation normalize → sample 8–16 frames → resize ~768px
   │
   ├─► Stage 1 — Vision  (Token Factory)
   │     NEBIUS_VISION_MODEL = openbmb/MiniCPM-V-4_5
   │     prompt: vision_observation.txt  · ≤10 images/request, chunked
   │     → per-phase bar-path observation (think-tags stripped)
   │
   ├─► Stage 2 — Judge  (Token Factory, NVIDIA open model)
   │     NEBIUS_JUDGE_MODEL = nvidia/Nemotron-3-Ultra-550b-a55b
   │     fallback: nvidia/nemotron-3-super-120b-a12b
   │     prompts: system.txt + user_snatch|clean|jerk.txt
   │     → Pydantic-validated JSON verdict
   │
   ├─► Annotated replay renderer (ffmpeg)
   │     trimmed clip + score/cue overlays → MP4 H.264 yuv420p faststart
   │
   └─► Result page (score, band, drift, confidence, cues, notes, replay, share)
```

**Key code:** `server/` holds the Python app (`services/nebius_client.py` is the Token Factory client for both stages, `services/pipeline.py` the run, `utils/frames.py` the sampler); `client/` holds everything served to the browser (`templates/`, `static/`); `prompts/` holds the model prompts (`system.txt`, `user_snatch.txt`, `user_clean.txt`, `user_jerk.txt`, `vision_observation.txt`, `vision_triage.txt`); `score_clip.py` and `score_session.py` are the CLIs; `tests/`.

**Deployment (Oct 20):** host the FastAPI app behind HTTPS (required for camera access and PWA install) and point it at Token Factory. Requests run as async jobs with polling so slow mobile connections don't time out.

### Environment variables
| Variable | Purpose | Default / example |
|---|---|---|
| `NEBIUS_API_KEY` | Token Factory API key (secret, never committed) | — |
| `NEBIUS_BASE_URL` | Token Factory endpoint | `https://api.tokenfactory.nebius.com/v1` |
| `NEBIUS_VISION_MODEL` | Stage 1 vision model | `openbmb/MiniCPM-V-4_5` |
| `NEBIUS_JUDGE_MODEL` | Stage 2 judge model | `nvidia/Nemotron-3-Ultra-550b-a55b` |
| `NEBIUS_JUDGE_FALLBACK_MODEL` | Judge fallback (empty disables) | `nvidia/nemotron-3-super-120b-a12b` |
| `NEBIUS_VISION_MAX_IMAGES` | Images per vision request | `10` |
| `NEBIUS_JUDGE_MAX_TOKENS` | Judge completion budget | `2048` |
| `MAX_UPLOAD_MB` | Upload ceiling; `0` disables | `200` |
| `JOB_TTL_SECONDS` | How long a job and its replay are kept | `1800` |

> All of the above are implemented and read in `nebius_client.py`, `uploads.py` and `jobs.py`, and are listed in `.env.example`.

## 14. Biomechanics basis

The rubric draws **only on principles** from Amit's Wingate course, quoted in our own words:
- **Keep the bar close to the body** through every phase
- **Apply force vertically**; horizontal bar travel is wasted effort and a sign of risk
- **Moment arms:** the farther the bar is from the body or base, the larger the moment arm and the harder the lift is to control

Course PDFs are **never committed** or redistributed.

## 15. Acceptance criteria & demo script

### Acceptance criteria
- [ ] **AC-1 (mobile):** a judge opens the demo URL **on a phone**, films or uploads a clip, and gets the score, cues and annotated video **without using a desktop**. The app is built and verified end to end at phone width in headless Chromium; **this stays open until the HTTPS deploy exists and a real phone has run it**.
- [x] AC-2: a side-view, trimmed **snatch** clip returns schema-valid JSON with every field filled and 1–2 cues. *(covered by `tests/test_api.py::test_single_lift_round_trip`; still to be run against a real clip.)*
- [x] AC-3: a clean clip works the same way, and a clean & jerk is split into two scored lifts. *(`tests/test_api.py::test_clean_and_jerk_comes_back_as_two_scored_lifts`.)*
- [x] AC-4: a portrait or non-side-view clip, or a mixed clean & jerk clip, returns **visibly lower confidence** and a warning. *(`server/services/confidence.py`; `tests/test_confidence.py` and `test_api.py::test_mixed_clip_that_cannot_be_split_lowers_confidence` pin the 0.85 case.)*
- [ ] AC-5: portrait phone video from iOS and Android is rotated correctly. **Needs real device clips** (rotation is read via ffprobe and applied; untested on real iOS/Android footage).
- [ ] AC-6: the annotated MP4 plays inline in iOS Safari and Android Chrome and can be downloaded or shared. **Needs real devices** (encoding is H.264/yuv420p/faststart and verified by `tests/test_replay.py`; inline playback verified in headless Chromium only).
- [x] AC-7: the Ultra → Super fallback works when the primary model fails. *(`tests/test_pipeline_mock.py::test_judge_falls_back_to_second_model_when_primary_fails`.)*
- [x] AC-8 (partly): tests pass and nothing new is committed — `samples/` and media extensions are gitignored. **The 43 MB `samples/clip.mp4` remains in git history at `225bf51`** by decision; purging it needs a force-push.
- [x] AC-9: all text in the UI and output is in English.

### Demo script (≤ 3-minute video)
1. **0:00–0:20 The problem:** "Bar path decides the lift, and you can't see it in real time."
2. **0:20–0:40 Mobile:** open the PWA from the phone's home screen, pick Snatch, see the filming guide.
3. **0:40–1:20 Capture:** film or upload a side-view snatch, trim it, watch the progress states.
4. **1:20–2:00 Result:** score, band, drift, confidence, cues; play the annotated replay and tap Share.
5. **2:00–2:20 Trust:** upload a poorly angled or mixed clip and show confidence drop with a warning.
6. **2:20–2:50 Under the hood:** the two-stage pipeline on Token Factory, with MiniCPM-V for vision and **NVIDIA Nemotron 3 Ultra** as the judge, plus Pydantic validation.
7. **2:50–3:00** Repo, MIT license, demo URL.

## 16. Roadmap

| Date | Milestone |
|---|---|
| ✅ Oct 14 | Two-stage pipeline verified (PRs #3, #4 merged); first real run |
| **Oct 17** | UI + cues/trust fixes: trim start/end, angle-aware confidence, side-view snatch test; **mobile-first layout + PWA manifest/icon + camera capture** |
| Oct 18–19 | Filming guide overlay, annotated MP4 replay (iOS-safe), share/download, real-device testing on iPhone and Android |
| **Oct 20** | Deploy over HTTPS; test the demo URL on phones over 4G |
| Oct 21–23 | Clean polish, mixed-lift detection, error states, latency tuning |
| **Oct 24** | Record the video (filmed on phone) and finish the README |
| **Oct 28** | **Submit** (two days of buffer) |
| Oct 30, 10:00 PDT / 19:00 IST | Hard deadline |
| Post-hackathon | Native app (React Native/Flutter); see §17 |

## 17. Risks & mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Vision model blends phases or lifts together | Wrong score | Trim to one lift, require side view, add phase-specific prompting, lower confidence when phases are unclear |
| Confidence is miscalibrated (0.85 on a bad clip) | Users trust bad scores | Angle- and quality-aware confidence rule (FR-8); label scores "indicative" in the UI |
| No Cosmos; NVIDIA models are text-only | Weaker Physical AI story | Enter Best Apps and Agents; make the Nemotron judge central in the README and video |
| Track fit gets questioned | Disqualification or a lower score | Check the rules text and confirm with the organizers before Oct 28 |
| Token Factory latency, rate limits or outages | Demo fails | Fallback judge, retries, async jobs, a pre-recorded backup demo clip |
| Invalid JSON from the judge | Crash | Pydantic validation, one repair retry, then a graceful error |
| iOS/HEVC/rotation quirks | Broken uploads or replays | See §9 mobile risks |
| Leaking secrets or media | Security or licensing problem | `.gitignore`, env-only secrets, a pre-commit check, never commit videos or PDFs |
| Scope creep (chat, Hebrew, accounts) | Missed deadline | Out-of-scope list (§3); hold the Oct 28 submit date |

## 18. Stretch goals

- Jerk support, plus clean & jerk auto-split into two scored calls
- A bar-path trace overlay drawn from per-frame bar positions on the annotated replay
- Comparing two attempts side by side
- **Native mobile app (React Native/Flutter):** 60 fps capture, locked exposure and focus, a live angle guide, on-device trimming, push notifications for finished results, an offline upload queue
- Re-check Cosmos or another NVIDIA vision model if one becomes available on Token Factory, to run fully on NVIDIA models

## 19. Submission checklist

- [ ] Public repo, MIT license visible (✅ PR #2)
- [ ] README: one-liner, mobile demo URL, screenshots, architecture, models used (NVIDIA Nemotron 3 Ultra on Token Factory), env vars, setup, filming guide, limitations
- [ ] Demo URL live over HTTPS and tested on an iPhone and an Android phone
- [ ] YouTube video ≤ 3 minutes, public or unlisted, linked in the README
- [ ] Track selected (recommended: **Best Apps and Agents**)
- [ ] Tool feedback form submitted
- [ ] No API keys, videos or course PDFs in git history
- [ ] Tests pass
- [ ] Submitted by **Oct 28** (deadline Oct 30, 10:00 PDT / 19:00 IST)
