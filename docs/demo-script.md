# Demo video — narration script

**For:** the ≤3-minute YouTube submission (Nebius × NVIDIA Global AI Hackathon).
**Length:** 403 spoken words ≈ **2:41 at 150 wpm**, inside the 3:00 ceiling with room to breathe
and to let two or three shots land without narration over them.
**Language:** English only, as everything else in this project.

Read the **Narration** aloud. The **On screen** note is what the viewer should be looking at while
you say it. Beat timings are targets, not a metronome — the word counts are what keep you honest.
Judging is on technical implementation, design, impact and idea; each beat is aimed at one, noted
in its heading.

---

## Before you record

- [ ] Demo URL live over HTTPS, opened once on the phone and **installed to the home screen**
      (the shot in beat 2 is the home-screen icon, not a browser tab).
- [ ] **Clip A — the good one:** a side-view snatch, landscape, 3–5 m, hip height, whole bar in
      frame. Leave a second either side of the attempt so the trim step has something to do.
- [ ] **Clip B — the honest one:** a clean & jerk in a single clip. This is the beat-5 shot.
- [ ] Both clips already in the phone's gallery. Do not rely on filming live during the take.
- [ ] **A screen recording of a full successful run**, captured beforehand. If Token Factory is
      slow or the venue wifi drops mid-take, cut to it rather than waiting on camera.
- [ ] Phone on Do Not Disturb, landscape screen recording, clean status bar.
- [ ] Know your real numbers — read the score, band and confidence the app actually returns.
      **Never narrate a number the screen does not show.**

---

## Script

### 1 · 0:00–0:20 — The problem · *impact*

> **On screen:** gym footage of a snatch, side on. Freeze at the transition and draw the bar's
> forward loop over the frozen frame. No UI yet.

> Every Olympic lift is decided by one thing: whether the bar stays close. But it happens in under
> three seconds, and from the side of the platform you can't see it. Frame-by-frame review takes
> time and an experienced eye. So most lifters never hear the one thing that would fix the lift.

*(52 words)*

---

### 2 · 0:20–0:40 — It's on the phone already in your hand · *design*

> **On screen:** phone home screen → tap the Lift Judge icon → it opens full-screen, no browser
> chrome → tap **Snatch** → the filming guide appears. Let the guide diagram sit on screen.

> Sideline Lift Judge installs to your home screen. No app store, no account. Pick the lift — and
> before the camera opens, it shows you how to film: square to the side, three to five metres
> back, hip height, whole bar in frame.

*(44 words)*

---

### 3 · 0:40–1:20 — Capture and trim · *technical implementation*

> **On screen:** tap **Upload from gallery**, choose clip A → the trim screen → drag the start and
> end handles, tap **Preview** so the selection loops → tap **Score this lift** → the staged
> progress list ticking through: finding the lift, reading the bar path, judging, rendering.

> I pick the clip and trim it to the single attempt. The trim is two timestamps — the cutting
> happens on the server, so nothing re-encodes on the phone.
>
> Then it tells you what it's doing at every step: finding the lift, reading the bar path, judging
> it, rendering the replay. That's a background job the page polls, so if the gym wifi drops, the
> result is still waiting when you come back. All of it runs on Nebius Token Factory.

*(81 words)*

---

### 4 · 1:20–2:00 — The verdict · *idea*

> **On screen:** the result screen. Hold on the big score. Scroll slowly: drift, confidence meter,
> the two cue cards. Then play the annotated replay inline and tap **Share** to open the share
> sheet. Let the share sheet be visible for a beat.

> Four out of ten. Poor. The bar drifted about a hand's width forward of mid-foot through the
> transition, worst just before contact.
>
> Then the part a coach can actually use: two cues, in the language you'd shout across the
> platform. Keep the bar close. Bar into the hips, contact, then explode up.
>
> And the verdict is burnt into the replay itself — so when I send this to my athlete, the score
> travels with the video.

*(76 words — adjust the first two sentences to the numbers on screen.)*

---

### 5 · 2:00–2:20 — The honest part · *impact*

> **On screen:** clip B. The result screen with the blue **"Two lifts in that clip"** banner and
> two separate verdict cards. Then cut to the mixed-clip case: confidence low, orange warning
> directly under the score. Hold on the warning.

> Now the honest part. This clip has a clean and a jerk in it. The app spots two lifts, splits
> them, and scores each on its own.
>
> When it can't separate them, confidence drops to thirty-five percent, with the warning right
> next to the score. An early version returned eighty-five percent on this clip. That was the bug
> worth fixing.

*(60 words)*

---

### 6 · 2:20–2:50 — Under the hood · *technical implementation*

> **On screen:** a simple two-box diagram — sampled frames → vision model → text description →
> Nemotron → validated JSON. Put the **NVIDIA Nemotron 3 Ultra** and **Nebius Token Factory** names
> on screen. Optionally flash the terminal running `score_clip.py --debug`.

> Two stages, both on Nebius Token Factory. A vision model turns the sampled frames into a plain
> description of where the bar went.
>
> Then NVIDIA's Nemotron 3 Ultra reasons over it: applies the rubric, scores the lift, picks the
> cues, and returns JSON validated against a strict schema — with a repair retry and a fallback
> judge.
>
> The rubric is weightlifting biomechanics: keep the bar close, drive force vertically, keep the
> moment arm short.

*(74 words)*

---

### 7 · 2:50–3:00 — Close

> **On screen:** the repo URL, the MIT badge and the demo URL, large and readable. End card.

> Public repo, MIT licensed, and a demo URL you can open on your own phone right now.

*(17 words)*

---

**Total: 403 spoken words ≈ 2:41 at 150 wpm.** If your read comes in long, cut in this order:

1. Beat 3 — the polling sentence (keep "runs on Nebius Token Factory"). *−34 words*
2. Beat 4 — the second cue. Show both on screen, say one. *−12 words*
3. Beat 6 — the biomechanics sentence. *−17 words*

That takes it to ~340 words ≈ 2:16, which leaves 40 seconds of footage to breathe.

---

## Claims check

Every line above is true of what the code does today. Do not add to it:

- The judge **is** an NVIDIA open model (Nemotron 3 Ultra) on Token Factory, with a Nemotron Super
  fallback.
- The vision stage is **not** an NVIDIA model — it is MiniCPM-V, because the Nemotron models on
  this account reject image input. If asked, say so plainly; the README says it too. Beat 6 is
  worded to be accurate about this: "a vision model", then NVIDIA named for the reasoning.
- Scores are **indicative**. There is no pose estimation and no pixel measurement of drift — the
  drift is a description, not a measurement. Don't call it one.
- Latency is "about a minute on a normal connection" — a target seen in testing, not a guarantee.
  Beat 3 deliberately makes no timing claim.
