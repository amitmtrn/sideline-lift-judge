/*
 * Sideline Pro — the paid coach review on top of the judge.
 *
 * Three roles, one page, no reload: the athlete buys a review under the
 * verdict (PayPal JS SDK, intent=authorize, so the money is held and not
 * charged) and reads its status; the coach works a queue of held reviews,
 * edits the AI draft and delivers (capture) or declines (void); ops talks
 * to the agent, which proposes cards that only a tap on Confirm executes.
 *
 * Everything from the server or the model is inserted with textContent,
 * never innerHTML. The sandbox banner is on every screen while Pro is on.
 */
(() => {
  "use strict";

  const CONFIG = window.LIFT_JUDGE_CONFIG || {};
  const PRO = CONFIG.pro || {};
  const ENABLED = Boolean(PRO.pro_enabled);
  const PRICE = Number(PRO.price_usd) || 15;
  const SLA_HOURS = Number(PRO.sla_hours) || 48;
  const COACH = PRO.coach_name || "Coach Maya";
  const TOKEN_KEY = "sidelineProAthleteToken";
  const NAME_KEY = "sidelineProAthleteName";
  const STATUS_POLL_MS = 8000;
  const DRAFT_POLL_MS = 2500;
  const DRAFT_POLL_MAX = 16;
  const HELD = new Set(["AUTHORIZED", "IN_REVIEW"]);

  const $ = (selector) => document.querySelector(selector);
  const screens = Array.from(document.querySelectorAll(".screen"));

  const state = {
    role: "athlete",
    sdk: null,
    pendingReview: null,
    statusTimer: null,
    draftTimer: null,
    currentCoachReview: null,
    actions: {},
  };

  /* ---------- helpers ---------- */

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text != null) node.textContent = text;
    return node;
  }

  function show(name) {
    screens.forEach((screen) => {
      screen.dataset.active = String(screen.dataset.screen === name);
    });
    window.scrollTo({ top: 0, behavior: "auto" });
    if (name !== "review-status") stopStatusPolling();
    if (name !== "coach-review") stopDraftPolling();
  }

  function money(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return "";
    return Number.isInteger(number) ? `$${number}` : `$${number.toFixed(2)}`;
  }

  function when(iso) {
    if (!iso) return "";
    const date = new Date(iso);
    if (Number.isNaN(date.getTime())) return iso;
    return date.toLocaleString(undefined, { weekday: "short", hour: "2-digit", minute: "2-digit", day: "numeric", month: "short" });
  }

  function countdown(seconds) {
    if (seconds == null) return "";
    if (seconds <= 0) return "past due";
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    if (hours >= 1) return `${hours} h ${String(minutes).padStart(2, "0")} m left`;
    return `${minutes} m left`;
  }

  function athleteToken() {
    try {
      let token = window.localStorage.getItem(TOKEN_KEY);
      if (!token) {
        const bytes = new Uint8Array(16);
        (window.crypto || window.msCrypto).getRandomValues(bytes);
        token = Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
        window.localStorage.setItem(TOKEN_KEY, token);
      }
      return token;
    } catch (error) {
      if (!state.fallbackToken) state.fallbackToken = String(Date.now()) + Math.random().toString(16).slice(2);
      return state.fallbackToken;
    }
  }

  function storedName() {
    try {
      return window.localStorage.getItem(NAME_KEY) || "";
    } catch (error) {
      return "";
    }
  }

  function rememberName(name) {
    try {
      if (name) window.localStorage.setItem(NAME_KEY, name);
    } catch (error) {
      /* private mode */
    }
  }

  async function api(path, options = {}) {
    const headers = { "X-Athlete-Token": athleteToken() };
    let body;
    if (options.body !== undefined) {
      headers["Content-Type"] = "application/json";
      body = JSON.stringify(options.body);
    }
    const response = await fetch(path, { method: options.method || "GET", headers, body, cache: "no-store" });
    let payload = {};
    try {
      payload = await response.json();
    } catch (error) {
      payload = {};
    }
    if (!response.ok) {
      const error = new Error(payload.detail || `Request failed (HTTP ${response.status}).`);
      error.status = response.status;
      error.retryAfter = Number(response.headers.get("Retry-After")) || 0;
      throw error;
    }
    return payload;
  }

  function notice(target, message, kind) {
    const node = typeof target === "string" ? $(target) : target;
    if (!node) return;
    node.hidden = !message;
    node.textContent = message || "";
    node.className = kind === "info" ? "notice hint" : "notice";
  }

  function statusBadge(review) {
    const badge = element("span", "status-badge", review.status_label || review.status);
    badge.dataset.state = HELD.has(review.status)
      ? "held"
      : review.status === "CAPTURED"
        ? "captured"
        : review.status === "REFUNDED"
          ? "refunded"
          : review.status === "VOIDED"
            ? "released"
            : "other";
    return badge;
  }

  /* ---------- roles and banner ---------- */

  function setRole(role) {
    state.role = role;
    document.querySelectorAll("#role-switch [data-role]").forEach((button) => {
      button.setAttribute("aria-pressed", String(button.dataset.role === role));
    });
    // #coach and #ops deep-link a role (handy on a second phone, or in the
    // demo video); the athlete's screens keep a clean URL.
    try {
      const hash = role === "athlete" ? "" : `#${role}`;
      window.history.replaceState(null, "", window.location.pathname + window.location.search + hash);
    } catch (error) {
      /* older browsers */
    }
    if (role === "coach") {
      show("coach");
      loadQueue();
    } else if (role === "ops") {
      show("ops");
      loadOps();
    } else {
      show("start");
      refreshMyReviewsButton();
    }
  }

  function initRoles() {
    const banner = $("#sandbox-banner");
    const nav = $("#role-switch");
    if (banner) banner.hidden = false;
    if (nav) nav.hidden = false;
    document.querySelectorAll("#role-switch [data-role]").forEach((button) => {
      button.addEventListener("click", () => setRole(button.dataset.role));
    });
    // Any athlete-side navigation (the judge's own buttons) snaps the switcher back.
    document.addEventListener("click", (event) => {
      const go = event.target.closest("[data-go]");
      if (go && state.role !== "athlete") {
        state.role = "athlete";
        document.querySelectorAll("#role-switch [data-role]").forEach((button) => {
          button.setAttribute("aria-pressed", String(button.dataset.role === "athlete"));
        });
      }
    });
  }

  /* ---------- the call to action under a verdict ---------- */

  function loadSdk() {
    if (state.sdk) return state.sdk;
    state.sdk = new Promise((resolve, reject) => {
      if (window.paypal && window.paypal.Buttons) {
        resolve(window.paypal);
        return;
      }
      const script = document.createElement("script");
      const params = new URLSearchParams({
        "client-id": PRO.paypal_client_id || "",
        intent: "authorize", // without this the SDK defaults to capture
        currency: "USD",
        components: "buttons",
        "disable-funding": "paylater,credit",
      });
      script.src = `https://www.paypal.com/sdk/js?${params.toString()}`;
      script.async = true;
      script.onload = () => (window.paypal ? resolve(window.paypal) : reject(new Error("PayPal SDK did not load")));
      script.onerror = () => reject(new Error("PayPal SDK could not be loaded. Check the connection and try again."));
      document.head.appendChild(script);
    });
    return state.sdk;
  }

  function decorateCard(card, item, context) {
    if (!ENABLED || !card || !item || !item.band) return;
    const cta = element("section", "pro-cta");
    cta.appendChild(element("h3", null, `Get a coach's eyes — ${money(PRICE)}`));
    cta.appendChild(
      element("p", "pro-cta-line", `Held, not charged, until ${COACH} delivers (${SLA_HOURS} h). Miss the deadline and the hold is released automatically.`)
    );
    const button = element("button", null, `Get a coach's eyes — ${money(PRICE)}`);
    button.type = "button";
    const panel = element("div", "checkout");
    panel.hidden = true;
    button.addEventListener("click", () => {
      button.hidden = true;
      panel.hidden = false;
      openCheckout(panel, item, context);
    });
    cta.appendChild(button);
    cta.appendChild(panel);
    card.appendChild(cta);
  }

  function openCheckout(panel, item, context) {
    panel.textContent = "";
    const nameField = element("div", "field");
    const nameLabel = element("label", null, "Your name (so the coach knows who)");
    nameLabel.htmlFor = `pro-name-${item.index}`;
    const nameInput = document.createElement("input");
    nameInput.type = "text";
    nameInput.id = `pro-name-${item.index}`;
    nameInput.maxLength = 80;
    nameInput.placeholder = "e.g. Dan";
    nameInput.value = storedName();
    nameInput.autocomplete = "name";
    nameField.appendChild(nameLabel);
    nameField.appendChild(nameInput);
    panel.appendChild(nameField);

    const noteField = element("div", "field");
    const noteLabel = element("label", null, "Anything the coach should know? (optional)");
    noteLabel.htmlFor = `pro-note-${item.index}`;
    const noteInput = document.createElement("textarea");
    noteInput.id = `pro-note-${item.index}`;
    noteInput.maxLength = 500;
    noteInput.rows = 2;
    noteInput.placeholder = "e.g. the catch always feels forward";
    noteField.appendChild(noteLabel);
    noteField.appendChild(noteInput);
    panel.appendChild(noteField);

    panel.appendChild(element("p", "field-hint", `${money(PRICE)} USD · PayPal sandbox · authorized now, captured only when ${COACH} delivers.`));
    const buttons = element("div", "paypal-buttons");
    panel.appendChild(buttons);
    const status = element("p", "notice hint");
    status.hidden = true;
    panel.appendChild(status);

    const say = (message, kind) => notice(status, message, kind);
    say("Loading PayPal…", "info");
    loadSdk()
      .then((paypal) => {
        say("", "info");
        paypal
          .Buttons({
            style: { layout: "vertical", color: "gold", shape: "rect", label: "pay", height: 48 },
            createOrder: async () => {
              const name = nameInput.value.trim();
              rememberName(name);
              say("Opening the checkout…", "info");
              const created = await api("/api/checkout/reviews", {
                method: "POST",
                body: {
                  source: context.source,
                  job_id: context.jobId,
                  lift_index: item.index,
                  athlete_name: name,
                  note: noteInput.value.trim(),
                  athlete_token: athleteToken(),
                },
              });
              state.pendingReview = created.review_id;
              return created.paypal_order_id;
            },
            onApprove: async () => {
              say("Placing the hold…", "info");
              const review = await api(`/api/checkout/reviews/${state.pendingReview}/approve`, { method: "POST" });
              state.pendingReview = null;
              openStatus(review);
            },
            onCancel: () => say("Checkout cancelled. Nothing was held.", "info"),
            onError: (error) => say((error && error.message) || "PayPal could not complete the checkout.", "error"),
          })
          .render(buttons)
          .catch((error) => say((error && error.message) || "PayPal buttons could not be shown.", "error"));
      })
      .catch((error) => say((error && error.message) || "PayPal could not be loaded.", "error"));
  }

  /* ---------- the athlete's status page ---------- */

  function stopStatusPolling() {
    if (state.statusTimer) {
      window.clearTimeout(state.statusTimer);
      state.statusTimer = null;
    }
  }

  async function openStatusById(reviewId) {
    try {
      const review = await api(`/api/reviews/${encodeURIComponent(reviewId)}`);
      openStatus(review);
    } catch (error) {
      show("review-status");
      const container = $("#review-status");
      container.textContent = "";
      container.appendChild(element("p", "notice", error.message || "That review could not be found."));
    }
  }

  function openStatus(review) {
    renderStatus(review);
    show("review-status");
    stopStatusPolling();
    if (HELD.has(review.status) || review.status === "CHECKOUT_CREATED") {
      state.statusTimer = window.setTimeout(() => openStatusById(review.id), STATUS_POLL_MS);
    }
  }

  function keyframeStrip(snapshot) {
    const frames = (snapshot && snapshot.keyframes) || [];
    if (!frames.length) return null;
    const strip = element("div", "keyframes");
    frames.forEach((frame) => {
      const figure = element("figure");
      const img = document.createElement("img");
      img.src = frame.url;
      img.alt = frame.label || "evidence frame";
      img.loading = "lazy";
      figure.appendChild(img);
      if (frame.label) figure.appendChild(element("figcaption", null, frame.label));
      strip.appendChild(figure);
    });
    return strip;
  }

  function aiReport(snapshot) {
    const block = element("div", "ai-report");
    const head = element("div", "score-row");
    const score = element("div", "score", snapshot.score == null ? "—" : String(snapshot.score));
    score.appendChild(element("span", "of", "/10"));
    head.appendChild(score);
    head.appendChild(element("span", "band", snapshot.band || "unknown"));
    block.appendChild(head);
    if (snapshot.drift) block.appendChild(element("p", "drift", `Max drift: ${snapshot.drift}`));
    const cues = (snapshot.evidenced_cues && snapshot.evidenced_cues.length ? snapshot.evidenced_cues : (snapshot.cues || []).map((text) => ({ text }))) || [];
    if (cues.length) {
      const list = element("ul", "cue-list");
      cues.forEach((cue) => {
        const entry = element("li");
        entry.appendChild(element("b", null, cue.text || ""));
        if (cue.why) entry.appendChild(document.createTextNode(` — ${cue.why}`));
        list.appendChild(entry);
      });
      block.appendChild(list);
    }
    if (snapshot.angle_warning) {
      const warning = element("p", "notice hint", `⚠ ${snapshot.angle_warning}`);
      block.appendChild(warning);
    }
    if (snapshot.confidence != null) block.appendChild(element("p", "field-hint", `AI confidence ${Math.round(snapshot.confidence * 100)}%`));
    return block;
  }

  function coachWords(review) {
    const words = review.coach_review;
    if (!words) return null;
    const block = element("section", "coach-words");
    block.appendChild(element("h3", null, `Reviewed by ${words.reviewed_by || COACH}${words.ai_assisted ? " (AI-assisted)" : ""}`));
    if (words.score != null || words.band) {
      const line = element("p", "override");
      const ai = review.snapshot || {};
      if (ai.score != null && words.score != null && ai.score !== words.score) {
        line.textContent = `AI ${ai.score}/10 → Coach ${words.score}/10${words.band ? ` (${words.band})` : ""}`;
      } else if (words.score != null) {
        line.textContent = `Coach's score ${words.score}/10${words.band ? ` (${words.band})` : ""}`;
      }
      if (line.textContent) block.appendChild(line);
    }
    if (words.summary) block.appendChild(element("p", "summary", words.summary));
    if (Array.isArray(words.cues) && words.cues.length) {
      const list = element("ol", "cue-list");
      words.cues.forEach((cue) => list.appendChild(element("li", null, cue.text || cue)));
      block.appendChild(list);
    }
    if (words.drill && words.drill.name) {
      block.appendChild(element("p", "drill", `Drill: ${words.drill.name}${words.drill.sets_reps ? ` · ${words.drill.sets_reps}` : ""}`));
    }
    return block;
  }

  function renderStatus(review) {
    const container = $("#review-status");
    container.textContent = "";
    const card = element("article", "review-card");
    const head = element("div", "review-head");
    head.appendChild(element("span", "lift-name", (review.snapshot && review.snapshot.lift) || "lift"));
    head.appendChild(statusBadge(review));
    card.appendChild(head);
    card.appendChild(element("p", "status-detail", review.status_detail || ""));
    if (HELD.has(review.status) && review.sla_due_at) {
      card.appendChild(element("p", "deadline", `Due by ${when(review.sla_due_at)} · ${countdown(review.seconds_to_deadline)}`));
    }
    const words = coachWords(review);
    if (words) card.appendChild(words);
    else if (HELD.has(review.status)) card.appendChild(element("p", "field-hint", `${COACH} will write the review here. This page refreshes by itself.`));
    if (review.snapshot) {
      const details = element("details");
      details.appendChild(element("summary", null, "The AI report the coach is working from"));
      details.appendChild(aiReport(review.snapshot));
      const frames = keyframeStrip(review.snapshot);
      if (frames) details.appendChild(frames);
      card.appendChild(details);
    }
    const ids = review.paypal || {};
    const money_line = element("p", "paypal-ids");
    money_line.appendChild(element("span", "k", "PayPal (sandbox)"));
    money_line.appendChild(
      document.createTextNode(
        [ids.order_id ? `order ${ids.order_id}` : "", ids.authorization_id ? `authorization ${ids.authorization_id}` : "", ids.capture_id ? `capture ${ids.capture_id}` : "", ids.refund_id ? `refund ${ids.refund_id}` : ""]
          .filter(Boolean)
          .join(" · ")
      )
    );
    card.appendChild(money_line);
    const link = element("p", "field-hint");
    link.appendChild(document.createTextNode("Status page: "));
    const anchor = element("a", null, `${window.location.origin}${review.status_url || `/review/${review.id}`}`);
    anchor.href = review.status_url || `/review/${review.id}`;
    link.appendChild(anchor);
    card.appendChild(link);
    container.appendChild(card);
  }

  async function openMyReviews() {
    show("my-reviews");
    const container = $("#my-reviews");
    container.textContent = "";
    container.appendChild(element("p", "field-hint", "Loading…"));
    try {
      const payload = await api("/api/reviews/mine");
      container.textContent = "";
      if (!payload.reviews.length) {
        container.appendChild(element("p", "notice hint", "No coach reviews on this device yet. Score a lift, then tap “Get a coach's eyes”."));
        return;
      }
      payload.reviews.forEach((review) => container.appendChild(reviewRow(review, () => openStatus(review))));
    } catch (error) {
      container.textContent = "";
      container.appendChild(element("p", "notice", error.message));
    }
  }

  async function refreshMyReviewsButton() {
    const actions = $("#my-reviews-actions");
    if (!actions) return;
    try {
      const payload = await api("/api/reviews/mine");
      actions.hidden = !payload.reviews.length;
    } catch (error) {
      actions.hidden = true;
    }
  }

  function reviewRow(review, onOpen, extra) {
    const row = element("button", "review-row");
    row.type = "button";
    const top = element("div", "row-top");
    const who = review.athlete_name ? `${review.athlete_name} · ` : "";
    top.appendChild(element("span", "lift-name", `${who}${(review.snapshot && review.snapshot.lift) || "lift"}`));
    top.appendChild(statusBadge(review));
    row.appendChild(top);
    const bottom = element("div", "row-bottom");
    const ai = review.snapshot || {};
    bottom.appendChild(element("span", null, ai.score != null ? `AI ${ai.score}/10 ${ai.band || ""}` : ""));
    if (HELD.has(review.status)) bottom.appendChild(element("span", "deadline", countdown(review.seconds_to_deadline)));
    else bottom.appendChild(element("span", null, when(review.delivered_at || review.updated_at)));
    row.appendChild(bottom);
    if (extra) row.appendChild(extra);
    row.addEventListener("click", onOpen);
    return row;
  }

  /* ---------- the coach ---------- */

  async function loadQueue() {
    const container = $("#coach-queue");
    notice("#coach-error", "");
    container.textContent = "";
    container.appendChild(element("p", "field-hint", "Loading the queue…"));
    try {
      const payload = await api("/api/coach/queue");
      container.textContent = "";
      const count = $("#coach-count");
      if (count) {
        count.hidden = !payload.held;
        count.textContent = payload.held ? String(payload.held) : "";
      }
      if (!payload.reviews.length) {
        container.appendChild(element("p", "notice hint", "Nothing in the queue. When an athlete pays for a review it appears here, nearest deadline first."));
        return;
      }
      payload.reviews.forEach((review) => container.appendChild(reviewRow(review, () => openCoachReview(review.id))));
    } catch (error) {
      container.textContent = "";
      notice("#coach-error", error.message);
    }
  }

  function stopDraftPolling() {
    if (state.draftTimer) {
      window.clearTimeout(state.draftTimer);
      state.draftTimer = null;
    }
  }

  async function openCoachReview(reviewId, attempt = 0) {
    stopDraftPolling();
    show("coach-review");
    const container = $("#coach-review");
    if (attempt === 0) {
      container.textContent = "";
      container.appendChild(element("p", "field-hint", "Opening…"));
    }
    let review;
    try {
      review = await api(`/api/coach/reviews/${encodeURIComponent(reviewId)}`);
    } catch (error) {
      container.textContent = "";
      container.appendChild(element("p", "notice", error.message));
      return;
    }
    state.currentCoachReview = review;
    renderCoachReview(review);
    if (!review.ai_draft && HELD.has(review.status) && attempt < DRAFT_POLL_MAX) {
      state.draftTimer = window.setTimeout(() => openCoachReview(reviewId, attempt + 1), DRAFT_POLL_MS);
    }
  }

  function textarea(id, label, value, rows) {
    const field = element("div", "field");
    const labelNode = element("label", null, label);
    labelNode.htmlFor = id;
    const input = document.createElement("textarea");
    input.id = id;
    input.rows = rows || 2;
    input.value = value || "";
    field.appendChild(labelNode);
    field.appendChild(input);
    return field;
  }

  function renderCoachReview(review) {
    const container = $("#coach-review");
    container.textContent = "";
    const snapshot = review.snapshot || {};
    const head = element("div", "review-head");
    head.appendChild(element("span", "lift-name", `${review.athlete_name || "Athlete"} · ${snapshot.lift || "lift"}`));
    head.appendChild(statusBadge(review));
    container.appendChild(head);
    if (HELD.has(review.status)) {
      container.appendChild(element("p", "deadline", `${money(review.price_usd)} held · due by ${when(review.sla_due_at)} · ${countdown(review.seconds_to_deadline)}`));
    } else {
      container.appendChild(element("p", "status-detail", review.status_detail || ""));
    }
    if (review.athlete_note) {
      const note = element("p", "athlete-note");
      note.appendChild(element("span", "k", "Athlete says"));
      note.appendChild(document.createTextNode(review.athlete_note));
      container.appendChild(note);
    }
    const frames = keyframeStrip(snapshot);
    if (frames) container.appendChild(frames);
    const report = element("details", "ai-panel");
    report.open = true;
    report.appendChild(element("summary", null, "AI report"));
    report.appendChild(aiReport(snapshot));
    container.appendChild(report);

    if (!HELD.has(review.status)) {
      const words = coachWords(review);
      if (words) container.appendChild(words);
      return;
    }

    const draft = review.ai_draft;
    const editor = element("form", "draft-editor");
    editor.addEventListener("submit", (event) => event.preventDefault());
    const label = element("p", "draft-label");
    label.appendChild(element("span", "tag", draft ? (draft.source === "model" ? "AI draft" : "Draft (template)") : "Draft"));
    label.appendChild(document.createTextNode(draft ? " — edit before delivering. The athlete sees only what you deliver." : " generating…"));
    editor.appendChild(label);
    if (!draft) {
      editor.appendChild(element("p", "field-hint", "The AI draft is being written from the report. This takes a few seconds."));
    }
    const summary = textarea("draft-summary", "Summary to the athlete", draft ? draft.summary : "", 4);
    editor.appendChild(summary);
    const cues = draft && Array.isArray(draft.cues) ? draft.cues : [];
    for (let position = 0; position < 3; position += 1) {
      const cue = cues[position];
      editor.appendChild(textarea(`draft-cue-${position}`, `Cue ${position + 1}${cue && cue.evidence_ref ? ` (from ${cue.evidence_ref})` : ""}${position > 0 ? " (optional)" : ""}`, cue ? cue.text : "", 2));
    }
    const drillRow = element("div", "pair-fields");
    drillRow.appendChild(textarea("draft-drill", "Drill", draft && draft.drill ? draft.drill.name : "", 2));
    drillRow.appendChild(textarea("draft-sets", "Sets × reps", draft && draft.drill ? draft.drill.sets_reps : "", 2));
    editor.appendChild(drillRow);
    if (draft && draft.confidence_note) editor.appendChild(element("p", "field-hint", `AI on its own confidence: ${draft.confidence_note}`));

    const overrideRow = element("div", "pair-fields");
    const scoreField = element("div", "field");
    const scoreLabel = element("label", null, `Score (AI said ${snapshot.score != null ? snapshot.score : "—"})`);
    scoreLabel.htmlFor = "draft-score";
    const scoreSelect = document.createElement("select");
    scoreSelect.id = "draft-score";
    for (let value = 1; value <= 10; value += 1) {
      const option = document.createElement("option");
      option.value = String(value);
      option.textContent = `${value}/10`;
      scoreSelect.appendChild(option);
    }
    const suggestedScore = (draft && draft.suggested_score) || snapshot.score || 5;
    scoreSelect.value = String(suggestedScore);
    scoreField.appendChild(scoreLabel);
    scoreField.appendChild(scoreSelect);
    overrideRow.appendChild(scoreField);
    const bandField = element("div", "field");
    const bandLabel = element("label", null, `Band (AI said ${snapshot.band || "—"})`);
    bandLabel.htmlFor = "draft-band";
    const bandSelect = document.createElement("select");
    bandSelect.id = "draft-band";
    ["elite", "solid", "mixed", "poor", "fail"].forEach((band) => {
      const option = document.createElement("option");
      option.value = band;
      option.textContent = band;
      bandSelect.appendChild(option);
    });
    bandSelect.value = (draft && draft.suggested_band) || snapshot.band || "mixed";
    bandField.appendChild(bandLabel);
    bandField.appendChild(bandSelect);
    overrideRow.appendChild(bandField);
    editor.appendChild(overrideRow);

    const error = element("p", "notice");
    error.hidden = true;
    editor.appendChild(error);
    const actions = element("div", "actions pair");
    const deliver = element("button", null, `Deliver · capture ${money(review.price_usd)}`);
    deliver.type = "button";
    deliver.disabled = !draft;
    const decline = element("button", "ghost", "Decline · release the hold");
    decline.type = "button";
    actions.appendChild(deliver);
    actions.appendChild(decline);
    editor.appendChild(actions);
    editor.appendChild(element("p", "field-hint", "Deliver captures the PayPal authorization and shows the athlete your words. Decline voids it: nothing is charged."));

    deliver.addEventListener("click", async () => {
      deliver.disabled = true;
      decline.disabled = true;
      notice(error, "");
      const body = {
        summary: $("#draft-summary").value.trim(),
        cues: [0, 1, 2].map((position) => ({ text: $(`#draft-cue-${position}`).value.trim() })).filter((cue) => cue.text),
        drill: { name: $("#draft-drill").value.trim(), sets_reps: $("#draft-sets").value.trim() },
        score: Number(scoreSelect.value),
        band: bandSelect.value,
      };
      try {
        const delivered = await api(`/api/coach/reviews/${encodeURIComponent(review.id)}/deliver`, { method: "POST", body });
        renderCoachReview(delivered);
        const done = element("p", "notice hint", `Delivered. ${money(delivered.price_usd)} captured (PayPal capture ${delivered.paypal.capture_id}). The athlete can read it now.`);
        $("#coach-review").insertBefore(done, $("#coach-review").firstChild);
        loadQueue();
      } catch (failure) {
        notice(error, failure.message);
        deliver.disabled = false;
        decline.disabled = false;
        if (failure.status === 409 || failure.status === 502) openCoachReview(review.id);
      }
    });
    decline.addEventListener("click", async () => {
      if (!window.confirm(`Decline this review and release the ${money(review.price_usd)} hold?`)) return;
      deliver.disabled = true;
      decline.disabled = true;
      try {
        const declined = await api(`/api/coach/reviews/${encodeURIComponent(review.id)}/decline`, { method: "POST" });
        renderCoachReview(declined);
        loadQueue();
      } catch (failure) {
        notice(error, failure.message);
        deliver.disabled = !draft;
        decline.disabled = false;
      }
    });
    container.appendChild(editor);
  }

  /* ---------- ops ---------- */

  async function loadOps() {
    notice("#ops-error", "");
    try {
      const overview = await api("/api/ops/overview");
      renderHeld(overview.held || []);
      renderLedger(overview.ledger || {});
      const examples = $("#ops-examples");
      examples.textContent = "";
      (overview.examples || []).forEach((text) => {
        const chip = element("button", "chip", text);
        chip.type = "button";
        chip.addEventListener("click", () => {
          $("#ops-prompt").value = text;
          $("#ops-prompt").focus();
        });
        examples.appendChild(chip);
      });
      const demo = $("#ops-demo-actions");
      if (demo) demo.hidden = !PRO.demo_mode;
      const hint = $("#ops-chat").querySelector(".chat-hint");
      if (!hint) {
        const first = element("p", "chat-hint field-hint", `Agent: ${overview.model || ""} on Nebius Token Factory · invoices via the ${overview.tool_source || "PayPal Agent Toolkit"} · ${overview.agent_calls_remaining_today == null ? "" : `${overview.agent_calls_remaining_today} model calls left today`}`);
        $("#ops-chat").appendChild(first);
      }
    } catch (error) {
      notice("#ops-error", error.message);
    }
    loadAudit();
  }

  function renderHeld(held) {
    const container = $("#ops-held");
    container.textContent = "";
    if (!held.length) {
      container.appendChild(element("p", "field-hint", "No holds right now."));
      return;
    }
    held.forEach((review) => {
      const extra = element("div", "row-actions");
      if (PRO.demo_mode) {
        const expire = element("button", "ghost small", "Expire now (demo)");
        expire.type = "button";
        expire.addEventListener("click", async (event) => {
          event.stopPropagation();
          if (!window.confirm("Pull this review's deadline to now? The sweeper will release the hold, as it would after 48 h.")) return;
          expire.disabled = true;
          try {
            await api(`/api/ops/reviews/${encodeURIComponent(review.id)}/expire`, { method: "POST", body: {} });
            addSystemLine(`Deadline reached for ${review.athlete_name || "the athlete"}'s ${review.snapshot.lift}: the ${money(review.price_usd)} hold was released (void). Not charged.`);
            loadOps();
          } catch (error) {
            notice("#ops-error", error.message);
            expire.disabled = false;
          }
        });
        extra.appendChild(expire);
      }
      container.appendChild(reviewRow(review, () => {}, extra));
    });
  }

  function renderLedger(ledger) {
    const container = $("#ops-ledger");
    container.textContent = "";
    const list = element("dl", "ledger");
    const add = (key, value) => {
      list.appendChild(element("dt", null, key));
      list.appendChild(element("dd", null, value));
    };
    add("Delivered", String(ledger.reviews_delivered || 0));
    add("Refunded", String(ledger.reviews_refunded || 0));
    add("Net earned", money(ledger.net_usd || 0));
    add("Holds open", `${ledger.holds_open || 0} (${money(ledger.held_usd || 0)})`);
    add("Expiring in 24 h", String(ledger.holds_expiring_within_24h || 0));
    container.appendChild(list);
    container.appendChild(element("p", "field-hint", "From the local ledger: PayPal's Transaction Search lags up to three hours."));
  }

  async function loadAudit() {
    const container = $("#ops-audit");
    try {
      const payload = await api("/api/ops/actions");
      container.textContent = "";
      if (!payload.actions.length) {
        container.appendChild(element("p", "field-hint", "No agent actions yet."));
        return;
      }
      payload.actions.forEach((action) => {
        const row = element("div", "audit-row");
        row.dataset.state = action.status;
        const top = element("div", "row-top");
        top.appendChild(element("span", "tool", action.tool));
        top.appendChild(element("span", "status-badge", action.status));
        row.appendChild(top);
        row.appendChild(element("p", "summary", action.summary || ""));
        const meta = [when(action.created_at), action.paypal_id ? `PayPal ${action.paypal_id}` : "", action.result && action.result.error ? `error: ${action.result.error}` : ""].filter(Boolean).join(" · ");
        row.appendChild(element("p", "field-hint", meta));
        container.appendChild(row);
      });
    } catch (error) {
      container.textContent = "";
      container.appendChild(element("p", "notice", error.message));
    }
  }

  function addBubble(role, text) {
    const chat = $("#ops-chat");
    const bubble = element("div", `bubble ${role}`);
    bubble.appendChild(element("p", null, text));
    chat.appendChild(bubble);
    bubble.scrollIntoView({ behavior: "smooth", block: "nearest" });
    return bubble;
  }

  function addSystemLine(text) {
    addBubble("system", text);
  }

  function actionCard(card) {
    const node = element("div", "action-card");
    node.dataset.state = card.status;
    node.appendChild(element("p", "action-title", card.tool === "invoice_athletes" ? "Proposed: invoices" : card.tool === "cancel_or_refund_review" ? (card.details && card.details.outcome === "refund" ? "Proposed: refund" : "Proposed: release hold") : "Proposed action"));
    node.appendChild(element("p", "action-summary", card.summary));
    const details = card.details || {};
    if (Array.isArray(details.invoices)) {
      const list = element("ul", "cue-list");
      details.invoices.forEach((line) => list.appendChild(element("li", null, `${line.athlete} — ${money(line.amount_usd)} — ${line.description}${line.email ? ` → ${line.email}` : ""}`)));
      node.appendChild(list);
    }
    node.appendChild(element("p", "field-hint", `Nothing has run. Expires ${when(card.expires_at)}.`));
    const actions = element("div", "actions pair");
    const confirm = element("button", null, "Confirm");
    confirm.type = "button";
    const reject = element("button", "ghost", "Reject");
    reject.type = "button";
    actions.appendChild(confirm);
    actions.appendChild(reject);
    node.appendChild(actions);
    const result = element("p", "action-result");
    result.hidden = true;
    node.appendChild(result);
    const finish = (text, stateName) => {
      actions.hidden = true;
      result.hidden = false;
      result.textContent = text;
      node.dataset.state = stateName;
      loadOps();
    };
    confirm.addEventListener("click", async () => {
      confirm.disabled = true;
      reject.disabled = true;
      try {
        const done = await api(`/api/ops/actions/${encodeURIComponent(card.id)}/confirm`, { method: "POST", body: { token: card.confirm_token } });
        const outcome = done.result || {};
        let text = `Done. ${done.summary || ""}`;
        if (outcome.outcome) text = `Done: ${outcome.outcome} · ${money(outcome.amount_usd)} · PayPal ${done.paypal_id || ""}`;
        if (Array.isArray(outcome.invoices)) {
          text = `Sent ${outcome.invoices.length} invoice${outcome.invoices.length === 1 ? "" : "s"}: ` + outcome.invoices.map((i) => `${i.athlete} ${i.invoice_id}`).join(", ") + (outcome.errors && outcome.errors.length ? ` · failed: ${outcome.errors.join("; ")}` : "");
          outcome.invoices.forEach((invoice) => {
            if (!invoice.link) return;
            const link = element("a", "invoice-link", `${invoice.athlete}: open invoice`);
            link.href = invoice.link;
            link.target = "_blank";
            link.rel = "noopener";
            node.appendChild(link);
          });
        }
        finish(text, "EXECUTED");
      } catch (error) {
        finish(`Not done: ${error.message}`, "FAILED");
      }
    });
    reject.addEventListener("click", async () => {
      confirm.disabled = true;
      reject.disabled = true;
      try {
        await api(`/api/ops/actions/${encodeURIComponent(card.id)}/reject`, { method: "POST", body: {} });
      } catch (error) {
        /* already gone */
      }
      finish("Rejected. Nothing ran.", "REJECTED");
    });
    return node;
  }

  async function ask(prompt) {
    const input = $("#ops-prompt");
    const send = $("#ops-send");
    const text = (prompt || "").trim();
    if (!text) return;
    notice("#ops-error", "");
    addBubble("user", text);
    input.value = "";
    send.disabled = true;
    const thinking = addBubble("assistant thinking", "Working… (looking things up, then proposing)");
    try {
      const reply = await api("/api/ops/agent", { method: "POST", body: { prompt: text } });
      thinking.remove();
      const bubble = addBubble("assistant", reply.reply || "");
      if (Array.isArray(reply.tools_called) && reply.tools_called.length) {
        bubble.appendChild(element("p", "tools-line", `Tools: ${reply.tools_called.join(", ")}`));
      }
      (reply.actions || []).forEach((card) => bubble.appendChild(actionCard(card)));
      bubble.scrollIntoView({ behavior: "smooth", block: "nearest" });
      loadAudit();
    } catch (error) {
      thinking.remove();
      notice("#ops-error", error.message + (error.retryAfter ? ` (try again in ${Math.ceil(error.retryAfter / 60)} min)` : ""));
    } finally {
      send.disabled = false;
    }
  }

  async function resetDemo() {
    if (!window.confirm("Reset the demo data? Every sandbox hold is released, reviews and actions are forgotten, the athletes re-seeded.")) return;
    try {
      const done = await api("/api/ops/reset", { method: "POST", body: {} });
      addSystemLine(`Demo reset: ${done.released_holds} hold${done.released_holds === 1 ? "" : "s"} released, rows cleared.`);
      loadOps();
      loadQueue();
    } catch (error) {
      notice("#ops-error", error.message);
    }
  }

  /* ---------- wiring ---------- */

  window.SidelinePro = { decorateCard, openStatusById, setRole };

  if (!ENABLED) {
    const nav = $("#role-switch");
    const banner = $("#sandbox-banner");
    if (nav) nav.hidden = true;
    if (banner) banner.hidden = true;
    return;
  }

  initRoles();
  refreshMyReviewsButton();

  document.addEventListener("click", (event) => {
    const target = event.target.closest("[data-pro]");
    if (!target) return;
    switch (target.dataset.pro) {
      case "my-reviews":
        openMyReviews();
        break;
      case "coach-queue":
        show("coach");
        loadQueue();
        break;
      case "reset-demo":
        resetDemo();
        break;
      default:
        break;
    }
  });

  const form = $("#ops-form");
  if (form) {
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      ask($("#ops-prompt").value);
    });
  }

  if (CONFIG.initialReview) {
    openStatusById(CONFIG.initialReview);
  } else if (window.location.hash.startsWith("#coach:")) {
    // #coach:<review id> opens one review straight in the editor. Read the
    // id first: setRole rewrites the hash.
    const reviewId = window.location.hash.slice("#coach:".length);
    setRole("coach");
    openCoachReview(reviewId);
  } else if (window.location.hash === "#coach" || window.location.hash === "#ops") {
    setRole(window.location.hash.slice(1));
  }
})();
