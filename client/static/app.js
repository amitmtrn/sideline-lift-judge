/*
 * Sideline Lift Judge — mobile flow.
 *
 * Start -> choose lift -> filming guide -> pick or film a clip -> trim -> poll
 * the job -> read the verdict; or Start -> the cached sample result in one tap.
 * Everything the model produced is inserted with textContent, never innerHTML:
 * the cues, notes and drift description are model output.
 */
(() => {
  "use strict";

  const CONFIG = window.LIFT_JUDGE_CONFIG || {};
  const MAX_UPLOAD_MB = CONFIG.maxUploadMb || 200;
  // Demo caps, all optional: 0 means no clip-length cap, false means no code.
  const MAX_CLIP_SECONDS = Number(CONFIG.maxClipSeconds) || 0;
  // open: no code anywhere. link: the code rode in on the URL and is sent
  // silently. code: it is typed on the first screen.
  const ACCESS_MODE = CONFIG.accessMode || "code";
  const ACCESS_REQUIRED = Boolean(CONFIG.accessRequired);
  const DEMO_DISABLED = Boolean(CONFIG.demoDisabled);
  const SAMPLE_AVAILABLE = Boolean(CONFIG.sampleAvailable);
  const ACCESS_CODE_KEY = "liftJudgeAccessCode";
  const POLL_INTERVAL_MS = 1400;
  const MAX_AUTO_RETRY_SECONDS = 60;
  const STAGE_ORDER = ["uploading", "splitting", "observing", "judging", "rendering"];
  const STAGE_INDEX = {
    uploading: 0,
    queued: 1,
    preparing: 1,
    splitting: 1,
    observing: 2,
    judging: 3,
    rendering: 4,
    done: 5,
  };
  // Refusals after which trying the same clip again cannot help today.
  const NO_RETRY_REASONS = new Set(["daily_limit", "disabled", "access_code", "too_long"]);

  const state = {
    lift: "snatch",
    file: null,
    duration: 0,
    start: 0,
    end: 0,
    trimAvailable: true,
    capHintShown: false,
    jobId: null,
    polling: null,
    previewTimer: null,
    objectUrl: null,
    retryTimer: null,
  };

  const $ = (selector) => document.querySelector(selector);
  const screens = Array.from(document.querySelectorAll(".screen"));
  const preview = $("#preview");
  const startSlider = $("#trim-start");
  const endSlider = $("#trim-end");
  const accessInput = $("#access-code");

  /* ---------- Access code ---------- */

  // Judges get a link with ?code=… in it. Keep the code on the device and take
  // it out of the address bar so it is not in every screenshot and share.
  function storedAccessCode() {
    try {
      return window.localStorage.getItem(ACCESS_CODE_KEY) || "";
    } catch (error) {
      return "";
    }
  }

  function rememberAccessCode(code) {
    try {
      if (code) window.localStorage.setItem(ACCESS_CODE_KEY, code);
      else window.localStorage.removeItem(ACCESS_CODE_KEY);
    } catch (error) {
      /* private mode: the field still holds it for this visit */
    }
  }

  function adoptAccessCodeFromUrl() {
    const url = new URL(window.location.href);
    const fromUrl = (url.searchParams.get("code") || "").trim();
    if (fromUrl) {
      rememberAccessCode(fromUrl);
      url.searchParams.delete("code");
      window.history.replaceState(null, "", url.pathname + (url.search || "") + url.hash);
    }
    if (accessInput) accessInput.value = fromUrl || storedAccessCode();
  }

  function currentAccessCode() {
    if (!ACCESS_REQUIRED) return "";
    // In link mode the field is hidden and the stored code is the only source.
    const code = ACCESS_MODE === "code" && accessInput ? accessInput.value.trim() : storedAccessCode();
    rememberAccessCode(code);
    return code;
  }

  // A refusal that is about getting in, not about the clip: shown on the lift
  // screen. In code mode the field opens for a retype; in link mode there is
  // nothing to type, so the way on is the link itself or the sample.
  function accessProblem(message) {
    show("lift");
    if (ACCESS_MODE === "code") {
      $("#access-field").hidden = false;
      if (accessInput) accessInput.focus();
    }
    fail($("#access-error"), message);
    $("#access-actions").hidden = !SAMPLE_AVAILABLE;
  }

  // R3: validate the code on "Next", before anyone films or uploads. Offline,
  // or with no code required, the check is skipped: the upload is checked
  // anyway, and a flaky connection must not block the way to the camera.
  async function verifyAccess() {
    if (DEMO_DISABLED) {
      accessProblem("Live scoring is paused right now. The sample result shows everything the app does.");
      return false;
    }
    if (!ACCESS_REQUIRED) return true;
    const code = currentAccessCode();
    if (!code) {
      accessProblem(
        ACCESS_MODE === "link"
          ? "This demo opens from the link on the submission page. Use that link, or see the sample result."
          : "Enter the access code first — it is on the submission page."
      );
      return false;
    }
    try {
      const response = await fetch("/api/access/check", {
        headers: { "X-Access-Code": code },
        cache: "no-store",
      });
      if (response.ok) return true;
      let payload = {};
      try {
        payload = await response.json();
      } catch (error) {
        payload = {};
      }
      accessProblem(payload.detail || "That access code is not right. Check it against the submission page.");
      return false;
    } catch (error) {
      return true;
    }
  }

  adoptAccessCodeFromUrl();

  /* ---------- Navigation ---------- */

  function show(name) {
    screens.forEach((screen) => {
      screen.dataset.active = String(screen.dataset.screen === name);
    });
    window.scrollTo({ top: 0, behavior: "instant" in window ? "instant" : "auto" });
    if (name !== "trim") stopPreview();
  }

  document.addEventListener("click", (event) => {
    const target = event.target.closest("[data-go], [data-pick], [data-action]");
    if (!target) return;
    if (target.dataset.go) {
      if (target.dataset.go === "lift" || target.dataset.go === "start") reset();
      show(target.dataset.go);
      return;
    }
    if (target.dataset.pick) {
      $(target.dataset.pick === "camera" ? "#camera-input" : "#gallery-input").click();
      return;
    }
    switch (target.dataset.action) {
      case "next":
        verifyAccess().then((ok) => {
          if (ok) {
            hide($("#access-error"));
            $("#access-actions").hidden = true;
            show("guide");
          }
        });
        break;
      case "preview-window":
        playWindow();
        break;
      case "score":
        submit();
        break;
      case "retry":
        submit();
        break;
      case "sample":
        loadSample();
        break;
    }
  });

  document.querySelectorAll('input[name="lift"]').forEach((radio) => {
    radio.addEventListener("change", () => {
      if (radio.checked) state.lift = radio.value;
    });
  });

  function reset() {
    stopPolling();
    stopPreview();
    cancelAutoRetry();
    state.file = null;
    state.jobId = null;
    state.duration = 0;
    state.start = 0;
    state.end = 0;
    if (state.objectUrl) {
      URL.revokeObjectURL(state.objectUrl);
      state.objectUrl = null;
    }
    preview.removeAttribute("src");
    preview.load();
    $("#results").textContent = "";
    hide($("#progress-error"));
    hide($("#pick-error"));
    hide($("#access-error"));
    hide($("#cap-warning"));
    $("#access-actions").hidden = true;
    $("#progress-actions").hidden = true;
  }

  function hide(element) {
    if (!element) return;
    element.hidden = true;
    element.textContent = "";
  }

  function fail(element, message) {
    element.hidden = false;
    element.textContent = message;
  }

  /* ---------- Picking a clip ---------- */

  ["#camera-input", "#gallery-input"].forEach((selector) => {
    $(selector).addEventListener("change", (event) => {
      const file = event.target.files && event.target.files[0];
      if (file) loadFile(file);
      event.target.value = "";
    });
  });

  function loadFile(file) {
    hide($("#pick-error"));
    setTrimAvailable(true);
    state.file = file;
    if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
    state.objectUrl = URL.createObjectURL(file);
    preview.src = state.objectUrl;

    const megabytes = file.size / (1024 * 1024);
    $("#file-size").textContent =
      megabytes < 1 ? `${Math.max(1, Math.round(file.size / 1024))} KB` : `${megabytes.toFixed(1)} MB`;
    if (megabytes > MAX_UPLOAD_MB) {
      fail(
        $("#size-warning"),
        `Clips need to be under ${MAX_UPLOAD_MB} MB${MAX_CLIP_SECONDS ? ` and ${MAX_CLIP_SECONDS} seconds` : ""}. This one is ${megabytes.toFixed(0)} MB — trim it to the lift in your gallery app first.`
      );
    } else if (megabytes > MAX_UPLOAD_MB / 2) {
      fail(
        $("#size-warning"),
        "Large clip — trimming to just the lift keeps the upload quick on gym wifi."
      );
    } else {
      hide($("#size-warning"));
    }
    show("trim");
  }

  preview.addEventListener("loadedmetadata", () => {
    // iOS Safari reports Infinity for a freshly captured clip until it seeks.
    const duration = Number.isFinite(preview.duration) ? preview.duration : 0;
    state.duration = duration;
    state.start = 0;
    state.end = duration;
    [startSlider, endSlider].forEach((slider) => {
      slider.min = "0";
      slider.max = String(duration || 1);
      slider.step = duration > 20 ? "0.1" : "0.05";
    });
    startSlider.value = "0";
    endSlider.value = String(duration || 1);
    if (!duration) {
      fail(
        $("#size-warning"),
        "Could not read this clip's length in the browser — it will be scored whole unless you trim it on the server."
      );
    } else if (MAX_CLIP_SECONDS && duration > MAX_CLIP_SECONDS + 0.05) {
      // Start inside the cap so the first thing the user does is slide to the
      // lift, not read a refusal.
      state.end = MAX_CLIP_SECONDS;
      endSlider.value = String(MAX_CLIP_SECONDS);
      state.capHintShown = true;
      fail(
        $("#size-warning"),
        `Clips need to be under ${MAX_CLIP_SECONDS} seconds, so the end handle starts at ${MAX_CLIP_SECONDS} s. Drag both handles to the one lift.`
      );
    }
    renderWindow();
  });

  // Some phone codecs (HEVC on Android Chrome, for one) will not play in the
  // browser even though ffmpeg on the server reads them fine. Keep the clip and
  // let it be scored whole — bouncing the user back would be a dead end.
  preview.addEventListener("error", () => {
    setTrimAvailable(false);
    fail(
      $("#size-warning"),
      "This browser cannot play that clip, so it cannot be trimmed here. It will be scored whole — film the lift on its own, or trim it in your gallery app first."
    );
    show("trim");
  });

  function setTrimAvailable(available) {
    state.trimAvailable = available;
    [startSlider, endSlider].forEach((slider) => {
      slider.disabled = !available;
    });
    const previewButton = document.querySelector('[data-action="preview-window"]');
    if (previewButton) previewButton.disabled = !available;
    document.querySelectorAll(".slider, .window-bar").forEach((node) => {
      node.hidden = !available;
    });
  }

  startSlider.addEventListener("input", () => {
    state.start = Number(startSlider.value);
    if (state.start > state.end - 0.3) {
      state.start = Math.max(0, state.end - 0.3);
      startSlider.value = String(state.start);
    }
    preview.currentTime = state.start;
    renderWindow();
  });

  endSlider.addEventListener("input", () => {
    state.end = Number(endSlider.value);
    if (state.end < state.start + 0.3) {
      state.end = Math.min(state.duration || state.end, state.start + 0.3);
      endSlider.value = String(state.end);
    }
    preview.currentTime = state.end;
    renderWindow();
  });

  function renderWindow() {
    $("#start-value").textContent = stamp(state.start);
    $("#end-value").textContent = stamp(state.end);
    const length = Math.max(0, state.end - state.start);
    $("#window-length").textContent = MAX_CLIP_SECONDS
      ? `${length.toFixed(1)} s selected · max ${MAX_CLIP_SECONDS} s`
      : `${length.toFixed(1)} s selected`;
    const total = state.duration || 1;
    const selection = $("#window-sel");
    selection.style.left = `${(state.start / total) * 100}%`;
    selection.style.width = `${Math.max(1, (length / total) * 100)}%`;

    // The server refuses a window over the cap; say so here, before the upload.
    const overCap = Boolean(MAX_CLIP_SECONDS) && state.trimAvailable && state.duration > 0 && length > MAX_CLIP_SECONDS + 0.05;
    const scoreButton = document.querySelector('[data-action="score"]');
    if (scoreButton) scoreButton.disabled = overCap;
    if (overCap) {
      if (state.capHintShown) {
        // The hint has done its job; let the warning take its place rather
        // than stack beneath it, under the sticky button.
        hide($("#size-warning"));
        state.capHintShown = false;
      }
      fail($("#cap-warning"), `${length.toFixed(1)} s selected — bring it under ${MAX_CLIP_SECONDS} s to score it.`);
    } else {
      hide($("#cap-warning"));
    }
  }

  function playWindow() {
    stopPreview();
    preview.currentTime = state.start;
    const play = preview.play();
    if (play && typeof play.catch === "function") play.catch(() => {});
    state.previewTimer = window.setInterval(() => {
      if (preview.currentTime >= state.end) {
        preview.pause();
        preview.currentTime = state.start;
        stopPreview();
      }
    }, 80);
  }

  function stopPreview() {
    if (state.previewTimer) {
      window.clearInterval(state.previewTimer);
      state.previewTimer = null;
    }
    if (!preview.paused) preview.pause();
  }

  function stamp(seconds) {
    const value = Math.max(0, Number(seconds) || 0);
    const minutes = Math.floor(value / 60);
    return `${minutes}:${(value - minutes * 60).toFixed(1).padStart(4, "0")}`;
  }

  /* ---------- Scoring ---------- */

  function submit() {
    cancelAutoRetry();
    if (!state.file) {
      show("guide");
      return;
    }
    if (DEMO_DISABLED) {
      showFailure("Live scoring is paused right now. The sample result shows everything the app does.", "disabled");
      return;
    }
    const accessCode = currentAccessCode();
    if (ACCESS_REQUIRED && !accessCode) {
      accessProblem(
        ACCESS_MODE === "link"
          ? "This demo opens from the link on the submission page. Use that link, or see the sample result."
          : "Enter the access code first — it is on the submission page."
      );
      return;
    }
    hide($("#progress-error"));
    $("#progress-actions").hidden = true;
    setStage("uploading", "Uploading the clip…", 0.02);
    show("progress");

    const body = new FormData();
    body.append("video", state.file, state.file.name || "clip.mp4");
    if (state.lift === "auto") {
      body.append("mode", "auto");
    } else {
      body.append("lift", state.lift);
      body.append("mode", "single");
    }
    // Only send a window if trimming worked and it is narrower than the clip.
    if (state.trimAvailable && state.duration && (state.start > 0.05 || state.end < state.duration - 0.05)) {
      body.append("trim_start", state.start.toFixed(2));
      body.append("trim_end", state.end.toFixed(2));
    }

    const request = new XMLHttpRequest();
    request.open("POST", "/api/jobs");
    if (accessCode) request.setRequestHeader("X-Access-Code", accessCode);
    request.upload.addEventListener("progress", (event) => {
      if (!event.lengthComputable) return;
      setStage("uploading", "Uploading the clip…", 0.02 + 0.16 * (event.loaded / event.total));
    });
    request.addEventListener("load", () => {
      let payload = {};
      try {
        payload = JSON.parse(request.responseText || "{}");
      } catch (error) {
        payload = {};
      }
      if (request.status >= 200 && request.status < 300 && payload.id) {
        state.jobId = payload.id;
        setStage("splitting", "Reading the clip…", 0.2);
        poll();
      } else if (payload.reason === "access_code") {
        accessProblem(payload.detail || "This demo needs an access code.");
      } else {
        const retryAfter = Number(request.getResponseHeader("Retry-After")) || 0;
        showFailure(payload.detail || `Upload failed (HTTP ${request.status}).`, payload.reason, retryAfter);
      }
    });
    request.addEventListener("error", () =>
      showFailure("The upload could not reach the server. Check your connection and try again.")
    );
    request.addEventListener("abort", () => showFailure("The upload was cancelled."));
    request.send(body);
  }

  function poll() {
    stopPolling();
    state.polling = window.setInterval(check, POLL_INTERVAL_MS);
    check();
  }

  function stopPolling() {
    if (state.polling) {
      window.clearInterval(state.polling);
      state.polling = null;
    }
  }

  async function check() {
    if (!state.jobId) return;
    let payload;
    try {
      const response = await fetch(`/api/jobs/${state.jobId}`, { cache: "no-store" });
      payload = await response.json();
      if (!response.ok) {
        stopPolling();
        showFailure(payload.detail || "That result is no longer available.");
        return;
      }
    } catch (error) {
      // A dropped poll is not fatal: the job keeps running on the server.
      return;
    }

    setStage(payload.stage, payload.message, payload.progress);
    if (payload.status === "done") {
      stopPolling();
      renderResults(payload);
      show("result");
    } else if (payload.status === "error") {
      stopPolling();
      showFailure(payload.error || payload.message || "Scoring failed.", payload.reason);
    }
  }

  function setStage(stage, message, fraction) {
    const index = STAGE_INDEX[stage] != null ? STAGE_INDEX[stage] : 1;
    STAGE_ORDER.forEach((name, position) => {
      const row = document.querySelector(`#stages li[data-stage="${name}"]`);
      if (!row) return;
      row.dataset.state = position < index ? "done" : position === index ? "active" : "";
    });
    if (message) $("#progress-note").textContent = message;
    const percent = Math.round(Math.max(0, Math.min(1, Number(fraction) || 0)) * 100);
    $("#progress-fill").style.width = `${percent}%`;
  }

  // One sentence and a way on. The reason decides which: a limit for the day
  // leaves only the sample; a busy server is retried by itself; anything
  // else offers the same clip again.
  function showFailure(message, reason, retryAfterSeconds) {
    stopPolling();
    cancelAutoRetry();
    fail($("#progress-error"), message);
    const actions = $("#progress-actions");
    actions.hidden = false;
    const retry = $("#progress-retry");
    if (retry) {
      retry.hidden = Boolean(reason) && NO_RETRY_REASONS.has(reason);
      retry.textContent = "Try again";
    }
    const sample = $("#progress-sample");
    if (sample) {
      // The sample leads when the clip cannot be scored today; it trails when
      // trying again is the natural next step.
      sample.classList.toggle("ghost", !(reason && NO_RETRY_REASONS.has(reason)));
    }
    show("progress");
    if (reason === "busy" && state.file) {
      scheduleAutoRetry(message, Math.min(MAX_AUTO_RETRY_SECONDS, Math.max(5, retryAfterSeconds || 30)));
    }
  }

  // "You're next": wait out the slot and send the same clip again, counting
  // down so the wait is visible, with the button there to jump the countdown.
  function scheduleAutoRetry(message, seconds) {
    let left = seconds;
    const note = $("#progress-error");
    const retry = $("#progress-retry");
    const tick = () => {
      note.textContent = `${message} Trying again in ${left} s…`;
      if (retry) retry.textContent = "Try again now";
      if (left <= 0) {
        cancelAutoRetry();
        submit();
        return;
      }
      left -= 1;
    };
    tick();
    state.retryTimer = window.setInterval(tick, 1000);
  }

  function cancelAutoRetry() {
    if (state.retryTimer) {
      window.clearInterval(state.retryTimer);
      state.retryTimer = null;
    }
  }

  /* ---------- The cached sample clip ---------- */

  // Scored once ahead of time and served from disk: no model call, no code,
  // not counted against any limit. What a judge sees when the budget is spent.
  async function loadSample() {
    stopPolling();
    cancelAutoRetry();
    state.jobId = null;
    hide($("#progress-error"));
    $("#progress-actions").hidden = true;
    setStage("done", "Loading the sample…", 1);
    show("progress");
    let payload;
    try {
      const response = await fetch("/api/sample", { cache: "no-store" });
      payload = await response.json();
      if (!response.ok) {
        showFailure(payload.detail || "The sample clip is not available right now.", "sample");
        return;
      }
    } catch (error) {
      showFailure("Could not load the sample clip. Check your connection and try again.", "sample");
      return;
    }
    renderResults(payload);
    show("result");
  }

  /* ---------- Rendering the verdict ---------- */

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text != null) node.textContent = text;
    return node;
  }

  function renderResults(payload) {
    const container = $("#results");
    container.textContent = "";
    const lifts = Array.isArray(payload.lifts) ? payload.lifts : [];

    if (payload.sample) {
      const banner = element("div", "split-banner");
      banner.appendChild(element("span", "icon", "▶"));
      const text = element("span");
      text.appendChild(element("b", null, "Sample result (pre-recorded). "));
      text.appendChild(
        document.createTextNode("Scored earlier and cached — nothing was sent to the model just now.")
      );
      banner.appendChild(text);
      container.appendChild(banner);
    }

    if (payload.split_detected) {
      const banner = element("div", "split-banner");
      banner.appendChild(element("span", "icon", "⚑"));
      const text = element("span");
      text.appendChild(element("b", null, "Two lifts in that clip. "));
      text.appendChild(
        document.createTextNode("It was split and each one scored on its own — a clean & jerk counts as two lifts.")
      );
      banner.appendChild(text);
      container.appendChild(banner);
    }

    const scored = lifts.filter((item) => item.band);
    if (!scored.length) {
      const notice = element(
        "p",
        "notice",
        lifts.length && lifts[0].error
          ? lifts[0].error
          : payload.message || "No snatch, clean or jerk could be scored in this clip."
      );
      container.appendChild(notice);
      appendSkipped(container, payload);
      return;
    }

    scored.forEach((item, position) => {
      container.appendChild(verdictCard(item, position));
    });
    // A lift that failed after others scored — the demo budget ran out
    // part-way, say — is shown, not silently dropped.
    lifts
      .filter((item) => !item.band && item.error)
      .forEach((item) => {
        const where =
          item.start_sec != null && item.end_sec != null ? ` ${stamp(item.start_sec)} – ${stamp(item.end_sec)}` : "";
        container.appendChild(element("p", "notice", `${item.lift || "lift"}${where}: ${item.error}`));
      });
    appendSkipped(container, payload);
  }

  function appendSkipped(container, payload) {
    if (!Array.isArray(payload.skipped) || !payload.skipped.length) return;
    const details = element("details");
    details.appendChild(element("summary", null, `Windows skipped (${payload.skipped.length})`));
    payload.skipped.forEach((line) => details.appendChild(element("p", null, line)));
    container.appendChild(details);
  }

  function verdictCard(item, position) {
    const card = element("article", "verdict");
    card.dataset.band = item.band || "unknown";
    card.style.animationDelay = `${position * 0.08}s`;

    const head = element("div", "verdict-head");
    head.appendChild(element("span", "lift-name", item.lift || ""));
    if (item.start_sec != null && item.end_sec != null) {
      head.appendChild(element("span", "window-stamp", `${stamp(item.start_sec)} – ${stamp(item.end_sec)}`));
    }
    card.appendChild(head);

    const row = element("div", "score-row");
    const score = element("div", "score", item.score == null ? "—" : String(item.score));
    score.appendChild(element("span", "of", "/10"));
    row.appendChild(score);
    row.appendChild(element("span", "band", item.band || "unknown"));
    card.appendChild(row);

    // The drift line is written on the server from the tracker's measurement
    // when there is one (the judge's words only lend the direction), so it
    // cannot contradict the number. Without a trace it is the judge's read.
    const drift = item.drift || (item.max_horizontal_drift ? { text: item.max_horizontal_drift, source: "judge" } : null);
    if (drift && drift.text) {
      const line = element("p", "drift");
      line.appendChild(
        element("span", "k", drift.source === "measurement" ? "Max drift, measured" : "Max drift, judge's read (bar not tracked)")
      );
      line.appendChild(document.createTextNode(drift.text));
      card.appendChild(line);
    }

    card.appendChild(pipelineLine(item));

    card.appendChild(confidenceBlock(item.confidence));

    if (Array.isArray(item.warnings) && item.warnings.length) {
      const list = element("ul", "warnings");
      item.warnings.forEach((warning) => {
        const entry = element("li");
        const informational = /scored on its own|separated out/i.test(warning);
        if (informational) entry.className = "info";
        entry.appendChild(element("span", "icon", informational ? "⚑" : "⚠"));
        entry.appendChild(element("span", null, warning));
        list.appendChild(entry);
      });
      card.appendChild(list);
    }

    if (Array.isArray(item.cues) && item.cues.length) {
      const cues = element("div", "cues");
      item.cues.forEach((cue, index) => {
        const block = element("div", "cue");
        block.appendChild(element("span", "tag", index === 0 ? "Cue" : "Then"));
        block.appendChild(document.createTextNode(cue));
        cues.appendChild(block);
      });
      card.appendChild(cues);
    }

    // The judge's own words on the drift stay readable when the measured line
    // replaced them above, so nothing the model said is hidden.
    const judgeDrift = drift && drift.source === "measurement" ? item.max_horizontal_drift : "";
    if (item.notes || judgeDrift) {
      const details = element("details");
      details.appendChild(element("summary", null, "Notes"));
      if (judgeDrift) details.appendChild(element("p", null, `Judge on the drift: ${judgeDrift}`));
      if (item.notes) details.appendChild(element("p", null, item.notes));
      card.appendChild(details);
    }

    if (item.observations) {
      const details = element("details");
      details.appendChild(element("summary", null, "Vision observations"));
      details.appendChild(element("pre", null, item.observations));
      card.appendChild(details);
    }

    if (item.annotated_video_url) {
      card.appendChild(replayBlock(item));
    }
    return card;
  }

  // Who did what for this verdict, as the pipeline reads: the local tracker
  // measured, the vision model observed, the judge judged. Names come from the
  // server, which knows which judge actually answered — the fallback, when
  // the primary failed — so the card never implies a model that was not used.
  function pipelineLine(item) {
    const models = item.models || {};
    const barPath = item.bar_path || {};
    const list = element("ol", "pipeline");
    let tracker = "CV tracker measured";
    if (barPath.tracked) {
      if (barPath.coverage != null) tracker += ` (plate seen in ${Math.round(barPath.coverage * 100)}% of frames)`;
    } else {
      tracker = "CV tracker: no trace, drift read from the frames";
    }
    list.appendChild(element("li", null, tracker));
    list.appendChild(element("li", null, `${models.vision_name || models.vision || "Vision model"} observed`));
    list.appendChild(
      element(
        "li",
        null,
        `${models.judge_name || models.judge || "Judge"} judged${models.judge_fallback ? " (fallback judge)" : ""}`
      )
    );
    return list;
  }

  function confidenceBlock(confidence) {
    const block = element("div", "confidence");
    const row = element("div", "row");
    row.appendChild(element("span", null, "Confidence"));
    row.appendChild(element("b", null, confidence == null ? "—" : `${Math.round(confidence * 100)}%`));
    block.appendChild(row);

    const meter = element("div", "meter");
    const level = confidence == null ? "low" : confidence < 0.45 ? "low" : confidence < 0.7 ? "medium" : "high";
    meter.dataset.level = level;
    const fill = element("span");
    fill.style.width = `${Math.round((confidence == null ? 0 : confidence) * 100)}%`;
    meter.appendChild(fill);
    block.appendChild(meter);
    return block;
  }

  function replayBlock(item) {
    const block = element("div", "replay");
    if (item.bar_path_tracked) {
      const key = element("p", "trace-key");
      key.appendChild(element("span", "swatch path", ""));
      key.appendChild(element("span", null, "bar path"));
      key.appendChild(element("span", "swatch start", ""));
      key.appendChild(element("span", null, "where it started"));
      key.appendChild(element("span", "swatch drift", ""));
      key.appendChild(element("span", null, "furthest from it"));
      block.appendChild(key);
    }
    const video = document.createElement("video");
    video.src = item.annotated_video_url;
    video.controls = true;
    video.playsInline = true;
    video.muted = true;
    video.preload = "metadata";
    video.setAttribute("playsinline", "");
    block.appendChild(video);

    const actions = element("div", "actions pair");
    const filename = `lift-judge-${item.lift || "lift"}-${item.score == null ? "na" : item.score}of10.mp4`;

    const download = element("a", "btn ghost", "⬇ Download");
    download.href = item.annotated_video_url;
    download.setAttribute("download", filename);
    actions.appendChild(download);

    if (navigator.share) {
      const share = element("button", "ghost", "↗ Share");
      share.type = "button";
      share.addEventListener("click", () => shareReplay(item, filename, share));
      actions.appendChild(share);
    }
    block.appendChild(actions);
    return block;
  }

  async function shareReplay(item, filename, button) {
    const label = button.textContent;
    button.disabled = true;
    button.textContent = "Preparing…";
    const title = `${item.lift} — ${item.score == null ? "—" : item.score}/10 (${item.band})`;
    const text = Array.isArray(item.cues) && item.cues.length ? item.cues[0] : "Bar-path score";
    try {
      const response = await fetch(item.annotated_video_url);
      const blob = await response.blob();
      const file = new File([blob], filename, { type: "video/mp4" });
      if (navigator.canShare && navigator.canShare({ files: [file] })) {
        await navigator.share({ files: [file], title, text });
      } else {
        // Sharing files is unsupported here; share the page instead.
        await navigator.share({ title, text, url: window.location.href });
      }
    } catch (error) {
      if (!error || error.name !== "AbortError") {
        button.textContent = "Share unavailable";
        window.setTimeout(() => {
          button.textContent = label;
          button.disabled = false;
        }, 2200);
        return;
      }
    }
    button.textContent = label;
    button.disabled = false;
  }

  /* ---------- PWA ---------- */

  if ("serviceWorker" in navigator) {
    window.addEventListener("load", () => {
      navigator.serviceWorker.register("/service-worker.js").catch(() => {});
    });
  }

  // iOS has no install prompt, so say so once.
  const standalone =
    window.matchMedia("(display-mode: standalone)").matches || window.navigator.standalone === true;
  const isIOS = /iphone|ipad|ipod/i.test(window.navigator.userAgent);
  if (isIOS && !standalone) {
    $("#install-hint").hidden = false;
  }

  renderWindow();
})();
