"""
The paid coach review over HTTP, end to end against a fake sandbox PayPal:
the sample → pay (authorize) → coach queue → AI draft → deliver (capture);
decline (void); the SLA sweep; refund; what each role may see; keyframes;
and that a review outlives the process.
"""
from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest

from server.services import reviews
from server.services.db import LiftSnapshot, ReviewOrder, session_scope, utcnow
from tests.pro_helpers import FakePayPal, buy_sample, paypal, pro_app, staff_login, staff_logout  # noqa: F401  (fixtures)


def _wait_drafts(client) -> None:
    from server.routes.checkout import wait_for_drafts

    # The TestClient runs the app loop in a thread; drafts are background
    # tasks on that loop, so give them a moment to land.
    client.portal.call(wait_for_drafts)


def _set_due(review_id: str, **delta) -> None:
    with session_scope() as session:
        review = session.get(ReviewOrder, review_id)
        review.sla_due_at = utcnow() + timedelta(**delta)


# ---------- pro off ----------


def test_without_paypal_keys_the_pro_routes_say_so(client):
    assert client.get("/api/config").json()["pro"]["pro_enabled"] is False
    response = client.post("/api/checkout/reviews", json={"source": "sample"})
    assert response.status_code == 503
    assert "not set up" in response.json()["detail"]
    assert client.get("/api/coach/queue").status_code == 503


def test_the_page_hides_the_flow_without_keys(client):
    html = client.get("/").text
    assert 'data-screen="coach"' in html  # the screens ship, the JS hides them
    assert "SANDBOX" in html


# ---------- pro on ----------


def test_config_exposes_the_public_bits_only(pro_app):
    pro = pro_app.get("/api/config").json()["pro"]
    assert pro["pro_enabled"] is True
    assert pro["paypal_client_id"] == "sb-client-id"
    assert pro["paypal_env"] == "sandbox"
    assert pro["price_usd"] == 15.0 and pro["sla_hours"] == 48.0
    assert pro["staff_login_configured"] is True
    assert "secret" not in str(pro).lower() and "test-pass" not in str(pro)


# ---------- the coach/ops sign-in ----------


STAFF_ROUTES = [
    ("GET", "/api/coach/queue"),
    ("GET", "/api/coach/reviews/{id}"),
    ("POST", "/api/coach/reviews/{id}/deliver"),
    ("POST", "/api/coach/reviews/{id}/decline"),
    ("POST", "/api/ops/agent"),
    ("POST", "/api/ops/actions/x/confirm"),
    ("POST", "/api/ops/actions/x/reject"),
    ("GET", "/api/ops/actions"),
    ("GET", "/api/ops/overview"),
    ("POST", "/api/ops/reviews/{id}/expire"),
    ("POST", "/api/ops/reset"),
]


def test_every_coach_and_ops_route_is_locked_without_the_sign_in(pro_app, paypal):
    """P0-2: server side, not just hidden in the UI; the athlete's own routes stay open."""
    review = buy_sample(pro_app, name="Dan")
    staff_logout(pro_app)
    for method, path in STAFF_ROUTES:
        response = pro_app.request(method, path.format(id=review["id"]), json={} if method == "POST" else None)
        assert response.status_code == 401, f"{method} {path}: {response.status_code} {response.text}"
        assert "sign in" in response.json()["detail"].lower()
    # Nothing moved while locked out.
    assert not any(call["method"] in ("capture_authorization", "void_authorization") for call in paypal.calls)
    # The athlete still reads their own review by its unguessable id, and can still buy.
    assert pro_app.get(f"/api/reviews/{review['id']}").status_code == 200
    assert pro_app.get("/api/reviews/mine", headers={"X-Athlete-Token": "tok-dan"}).json()["reviews"][0]["id"] == review["id"]
    assert pro_app.post("/api/checkout/reviews", json={"source": "sample", "athlete_token": "t2"}).status_code == 201
    # Signed in, the same routes answer.
    staff_login(pro_app)
    assert pro_app.get("/api/coach/queue").status_code == 200
    assert pro_app.get("/api/ops/overview").status_code == 200


def test_sign_in_is_a_signed_httponly_cookie(pro_app, monkeypatch):
    staff_logout(pro_app)
    me = pro_app.get("/api/auth/me").json()
    assert me == {"staff": False, "role": None, "configured": True}
    wrong = pro_app.post("/api/auth/login", json={"password": "nope"})
    assert wrong.status_code == 401 and "sp_staff" not in wrong.headers.get("set-cookie", "")
    assert pro_app.get("/api/coach/queue").status_code == 401

    ok = staff_login(pro_app)
    cookie = ok.headers["set-cookie"]
    assert "sp_staff=" in cookie and "HttpOnly" in cookie and "SameSite=lax" in cookie.replace("Lax", "lax")
    assert pro_app.get("/api/auth/me").json()["staff"] is True
    assert pro_app.get("/api/coach/queue").status_code == 200

    # A forged or tampered cookie is worthless.
    token = pro_app.cookies.get("sp_staff")
    role, expires, nonce, signature = token.split(".")
    pro_app.cookies.set("sp_staff", f"{role}.{int(expires) + 10}.{nonce}.{signature}")
    assert pro_app.get("/api/coach/queue").status_code == 401
    pro_app.cookies.set("sp_staff", "staff.9999999999.abcd.notasignature")
    assert pro_app.get("/api/coach/queue").status_code == 401

    # Changing the password signs everyone out; no password locks the routes with a reason.
    pro_app.cookies.set("sp_staff", token)
    monkeypatch.setenv("COACH_PASSWORD", "another-one")
    assert pro_app.get("/api/coach/queue").status_code == 401
    monkeypatch.setenv("COACH_PASSWORD", "")
    locked = pro_app.get("/api/ops/overview")
    assert locked.status_code == 401 and "COACH_PASSWORD" in locked.json()["detail"]
    assert pro_app.post("/api/auth/login", json={"password": "x"}).status_code == 503
    assert pro_app.get("/api/auth/me").json()["configured"] is False


def test_wrong_passwords_are_rate_limited_per_address(pro_app, monkeypatch):
    staff_logout(pro_app)
    monkeypatch.setenv("STAFF_LOGIN_HOURLY_LIMIT", "3")
    for _ in range(3):
        assert pro_app.post("/api/auth/login", json={"password": "nope"}).status_code == 401
    blocked = pro_app.post("/api/auth/login", json={"password": "nope"})
    assert blocked.status_code == 429 and "Retry-After" in blocked.headers
    # The right password is blocked too, until the hour turns: the limit is on attempts.
    assert pro_app.post("/api/auth/login", json={"password": "test-pass"}).status_code == 429


def test_buying_the_sample_holds_the_money_and_snapshots_the_lift(pro_app, paypal):
    created = pro_app.post("/api/checkout/reviews", json={"source": "sample", "athlete_token": "tok-dan", "athlete_name": "Dan"})
    assert created.status_code == 201
    body = created.json()
    assert body["status"] == "CHECKOUT_CREATED" and body["paypal_order_id"] == "ORDER001"
    assert paypal.calls[0]["method"] == "create_order"
    assert paypal.calls[0]["amount_cents"] == 1500 and paypal.calls[0]["custom_id"] == body["review_id"]
    assert paypal.calls[0]["request_id"] == f"create-{body['review_id']}"

    approved = pro_app.post(f"/api/checkout/reviews/{body['review_id']}/approve").json()
    assert approved["status"] == "AUTHORIZED"
    assert approved["paypal"]["authorization_id"] == "AUTH002"
    assert approved["sla_due_at"] and approved["seconds_to_deadline"] > 47 * 3600
    assert "Held, not charged" in approved["status_detail"]
    assert approved["coach_review"] is None and "ai_draft" not in approved
    snapshot = approved["snapshot"]
    assert snapshot["lift"] in ("snatch", "clean", "jerk") and snapshot["score"] is not None
    assert snapshot["keyframes"], "the sample's evidence stills were copied"
    image = pro_app.get(snapshot["keyframes"][0]["url"])
    assert image.status_code == 200 and image.headers["content-type"] == "image/jpeg"
    assert len(image.content) > 1000


def test_approve_is_idempotent(pro_app, paypal):
    review = buy_sample(pro_app)
    again = pro_app.post(f"/api/checkout/reviews/{review['id']}/approve")
    assert again.status_code == 200 and again.json()["status"] == "AUTHORIZED"
    assert sum(1 for call in paypal.calls if call["method"] == "authorize_order") == 1


def test_the_coach_sees_the_review_with_a_draft_and_delivers(pro_app, paypal):
    review = buy_sample(pro_app, name="Dan")
    _wait_drafts(pro_app)
    queue = pro_app.get("/api/coach/queue").json()
    assert queue["held"] == 1
    row = queue["reviews"][0]
    assert row["id"] == review["id"] and row["athlete_name"] == "Dan" and row["status"] == "AUTHORIZED"
    assert row["has_draft"] is True

    detail = pro_app.get(f"/api/coach/reviews/{review['id']}").json()
    assert detail["status"] == "IN_REVIEW", "opening it marks it in review"
    draft = detail["ai_draft"]
    assert draft["source"] == "template" and draft["cues"] and draft["summary"]
    assert detail["draft_status"] == "template"
    assert detail["snapshot"]["evidenced_cues"] or detail["snapshot"]["cues"]

    delivered = pro_app.post(
        f"/api/coach/reviews/{review['id']}/deliver",
        json={"summary": draft["summary"] + " Keep it close.", "cues": draft["cues"][:2], "drill": draft["drill"], "score": 7, "band": "solid"},
    )
    assert delivered.status_code == 200, delivered.text
    body = delivered.json()
    assert body["status"] == "CAPTURED" and body["paypal"]["capture_id"] == "CAP003"
    assert body["coach_review"]["score"] == 7 and body["coach_review"]["draft_kept_ratio"] > 0.5
    assert body["coach_review"]["review_seconds"] is not None
    captures = [call for call in paypal.calls if call["method"] == "capture_authorization"]
    assert len(captures) == 1 and captures[0]["request_id"] == f"capture-{review['id']}-1"

    # The athlete now reads the coach's words, and the money line says charged.
    mine = pro_app.get(f"/api/reviews/{review['id']}").json()
    assert mine["status"] == "CAPTURED" and mine["review_visible"] is True
    assert "charged" in mine["status_label"].lower()
    assert mine["coach_review"]["reviewed_by"] == "Coach Maya" and mine["coach_review"]["ai_assisted"] is True
    assert "ai_draft" not in mine


def test_the_athlete_never_sees_words_before_capture(pro_app):
    review = buy_sample(pro_app)
    with session_scope() as session:
        row = session.get(ReviewOrder, review["id"])
        row.coach_review_json = '{"summary":"early"}'
    mine = pro_app.get(f"/api/reviews/{review['id']}").json()
    assert mine["coach_review"] is None and mine["review_visible"] is False


def test_decline_releases_the_hold(pro_app, paypal):
    review = buy_sample(pro_app)
    declined = pro_app.post(f"/api/coach/reviews/{review['id']}/decline").json()
    assert declined["status"] == "VOIDED" and declined["void_reason"] == "coach_declined"
    assert any(call["method"] == "void_authorization" for call in paypal.calls)
    mine = pro_app.get(f"/api/reviews/{review['id']}").json()
    assert "not charged" in mine["status_detail"].lower() and "declined" in mine["status_detail"]
    # Nothing can be delivered on a released hold.
    late = pro_app.post(f"/api/coach/reviews/{review['id']}/deliver", json={"summary": "x"})
    assert late.status_code == 409


def test_the_sla_sweep_voids_an_expired_hold_and_capture_is_refused(pro_app, paypal):
    review = buy_sample(pro_app)
    _set_due(review["id"], seconds=-1)
    # Deliver after the deadline: refused, no capture call.
    late = pro_app.post(f"/api/coach/reviews/{review['id']}/deliver", json={"summary": "too late"})
    assert late.status_code == 409 and "window has passed" in late.json()["detail"]
    assert not any(call["method"] == "capture_authorization" for call in paypal.calls)
    # The status page sweeps: the hold is gone.
    mine = pro_app.get(f"/api/reviews/{review['id']}").json()
    assert mine["status"] == "VOIDED" and mine["void_reason"] == "sla_expired"
    assert mine["status_detail"] == "Coach missed the deadline. Your $15 hold was released. You were not charged."


def test_expire_now_is_the_demo_shortcut(pro_app, paypal):
    review = buy_sample(pro_app)
    expired = pro_app.post(f"/api/ops/reviews/{review['id']}/expire")
    assert expired.status_code == 200, expired.text
    assert expired.json()["status"] == "VOIDED" and expired.json()["void_reason"] == "sla_expired"


def test_expire_now_is_for_signed_in_staff_in_demo_mode_only(pro_app, monkeypatch, paypal):
    review = buy_sample(pro_app)
    staff_logout(pro_app)
    assert pro_app.post(f"/api/ops/reviews/{review['id']}/expire").status_code == 401
    assert pro_app.post("/api/ops/reset").status_code == 401
    staff_login(pro_app)
    monkeypatch.setenv("PRO_DEMO_MODE", "false")
    assert pro_app.post(f"/api/ops/reviews/{review['id']}/expire").status_code == 403
    assert pro_app.post("/api/ops/reset").status_code == 403
    assert not any(call["method"] == "void_authorization" for call in paypal.calls)
    monkeypatch.setenv("PRO_DEMO_MODE", "true")
    assert pro_app.post(f"/api/ops/reviews/{review['id']}/expire").status_code == 200


def test_a_transient_capture_error_is_retried_once(pro_app, paypal):
    review = buy_sample(pro_app)
    paypal.capture_failures = 1
    delivered = pro_app.post(f"/api/coach/reviews/{review['id']}/deliver", json={"summary": "ok"})
    assert delivered.status_code == 200 and delivered.json()["status"] == "CAPTURED"
    captures = [call for call in paypal.calls if call["method"] == "capture_authorization"]
    assert [c["request_id"] for c in captures] == [f"capture-{review['id']}-1", f"capture-{review['id']}-2"]


def test_a_hard_capture_failure_releases_the_hold(pro_app, paypal):
    review = buy_sample(pro_app)
    paypal.capture_failures = 2
    delivered = pro_app.post(f"/api/coach/reviews/{review['id']}/deliver", json={"summary": "ok"})
    assert delivered.status_code == 502
    assert "not charged" in delivered.json()["detail"]
    mine = pro_app.get(f"/api/reviews/{review['id']}").json()
    assert mine["status"] == "VOIDED" and mine["void_reason"] == "capture_failed"
    assert any(call["method"] == "void_authorization" for call in paypal.calls)


def test_a_review_survives_a_restart(pro_app, tmp_path, monkeypatch, paypal):
    """P0-3: the store is a file, not the process. A new app over the same file finds the review."""
    from fastapi.testclient import TestClient

    from server.main import app
    from server.services.db import reset_db_for_tests

    review = buy_sample(pro_app, name="Noa")
    _wait_drafts(pro_app)
    reset_db_for_tests()  # drop the engine, as a restart would
    with TestClient(app) as restarted:
        staff_login(restarted)
        queue = restarted.get("/api/coach/queue").json()
        assert [row["id"] for row in queue["reviews"]] == [review["id"]]
        detail = restarted.get(f"/api/coach/reviews/{review['id']}").json()
        assert detail["ai_draft"] is not None and detail["snapshot"]["keyframes"]
        assert detail["athlete_name"] == "Noa"


def test_mine_lists_this_devices_reviews_only(pro_app):
    buy_sample(pro_app, name="Dan", token="tok-a")
    buy_sample(pro_app, name="Noa", token="tok-b")
    mine = pro_app.get("/api/reviews/mine", headers={"X-Athlete-Token": "tok-a"}).json()["reviews"]
    assert [row["athlete_name"] for row in mine] == ["Dan"]
    assert pro_app.get("/api/reviews/mine").json() == {"reviews": []}


def test_the_status_page_route_opens_the_app_on_that_review(pro_app):
    review = buy_sample(pro_app)
    html = pro_app.get(f"/review/{review['id']}").text
    assert f'initialReview: "{review["id"]}"' in html


def test_abandoned_checkouts_are_swept(pro_app):
    created = pro_app.post("/api/checkout/reviews", json={"source": "sample", "athlete_token": "t"}).json()
    with session_scope() as session:
        row = session.get(ReviewOrder, created["review_id"])
        row.created_at = utcnow() - timedelta(minutes=31)
    queue = pro_app.get("/api/coach/queue").json()
    mine = pro_app.get(f"/api/reviews/{created['review_id']}").json()
    assert mine["status"] == "ABANDONED"
    # Neither the unpaid checkout nor its abandoned remains clutter the coach's queue.
    assert queue["reviews"] == []
    assert pro_app.get("/api/coach/queue").json()["reviews"] == []
    assert pro_app.get("/api/ops/overview").json()["held"] == []


def test_a_job_result_can_be_bought_too(pro_app, tmp_path, monkeypatch):
    """A live result is snapshotted from the job store before the job dies."""
    from server.services.jobs import get_store

    store = get_store()
    job = store.create()
    job.status = "done"
    job.payload = {
        "lifts": [
            {"index": 1, "lift": "clean", "score": 5, "band": "mixed", "cues": ["finish the pull"], "confidence": 0.6, "warnings": ["Camera looks off-axis: drift may be overstated."], "evidenced_cues": [], "bar_path": {"tracked": True, "max_drift_cm_estimate": 9}}
        ]
    }
    created = pro_app.post("/api/checkout/reviews", json={"source": "job", "job_id": job.id, "lift_index": 1, "athlete_token": "t"})
    assert created.status_code == 201, created.text
    review = pro_app.post(f"/api/checkout/reviews/{created.json()['review_id']}/approve").json()
    assert review["snapshot"]["lift"] == "clean" and review["snapshot"]["angle_warning"].startswith("Camera")
    assert review["snapshot"]["measurement"]["max_drift_cm_estimate"] == 9
    gone = pro_app.post("/api/checkout/reviews", json={"source": "job", "job_id": "nope", "athlete_token": "t"})
    assert gone.status_code == 404 and "expired" in gone.json()["detail"]


def test_the_lift_is_stored_as_a_key_and_shown_as_a_word(pro_app):
    """P1-2: a live job carries the LiftType enum; "LiftType.snatch" must never reach a screen."""
    from server.schemas.judgment import LiftType
    from server.services.jobs import get_store

    store = get_store()
    job = store.create()
    job.status = "done"
    job.payload = {"lifts": [{"index": 1, "lift": LiftType.snatch, "score": 6, "band": "mixed", "cues": ["x"], "confidence": 0.7, "warnings": [], "evidenced_cues": []}]}
    created = pro_app.post("/api/checkout/reviews", json={"source": "job", "job_id": job.id, "lift_index": 1, "athlete_token": "t", "athlete_name": "Amit"})
    review = pro_app.post(f"/api/checkout/reviews/{created.json()['review_id']}/approve").json()
    assert review["snapshot"]["lift"] == "snatch" and review["snapshot"]["lift_label"] == "Snatch"
    row = pro_app.get("/api/coach/queue").json()["reviews"][0]
    assert row["snapshot"]["lift"] == "snatch" and row["snapshot"]["lift_label"] == "Snatch"
    assert "LiftType" not in pro_app.get("/api/coach/queue").text
    assert "LiftType" not in pro_app.get(f"/api/coach/reviews/{review['id']}").text

    # A row written before the fix reads the same way, and the agent's lookup still matches it.
    with session_scope() as session:
        snapshot = session.get(LiftSnapshot, review["snapshot"]["id"])
        snapshot.lift_type = "LiftType.snatch"
    overview = pro_app.get("/api/ops/overview").json()
    assert overview["held"][0]["snapshot"] == {"lift": "snatch", "lift_label": "Snatch", "score": 6, "band": "mixed"}
    with session_scope() as session:
        assert [r.id for r in reviews.find_reviews(session, lift="snatch")] == [review["id"]]
        assert reviews.find_reviews(session, lift="clean") == []
    assert "LiftType." not in pro_app.get("/").text


def test_live_mode_is_refused(monkeypatch):
    from server.services import pro_config

    monkeypatch.setenv("PAYPAL_ENV", "live")
    with pytest.raises(pro_config.ProConfigError):
        pro_config.assert_sandbox()
    monkeypatch.setenv("PAYPAL_ENV", "sandbox")
    pro_config.assert_sandbox()
