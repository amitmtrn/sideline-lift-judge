"""
The access-code input must never render the secret in a visible form.
The template serves type=password + autocomplete=off and ships a hidden
"Access code saved ✓" line the JS swaps in once a code is remembered on
the device, so a judge coming from the ?code= link never sees the field
at all. These are DOM-level guarantees, so test them from the HTML.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from server.main import app


def _page(monkeypatch) -> str:
    monkeypatch.setenv("NEBIUS_API_KEY", "sk-test")
    monkeypatch.setenv("DEMO_ACCESS_CODE", "judge-1")
    monkeypatch.setenv("ACCESS_MODE", "code")
    with TestClient(app) as client:
        response = client.get("/")
    assert response.status_code == 200
    return response.text


def test_access_input_is_password_with_autocomplete_off(monkeypatch):
    html = _page(monkeypatch)
    # Find the whole <input id="access-code" ...> block.
    start = html.index('id="access-code"')
    end = html.index(">", start)
    tag = html[start:end]
    assert 'type="password"' in tag, tag
    assert 'autocomplete="off"' in tag, tag


def test_saved_row_is_shipped_hidden_so_js_can_show_it(monkeypatch):
    html = _page(monkeypatch)
    # Hidden by default; the JS un-hides it once a stored code is detected.
    assert 'id="access-saved"' in html
    saved_index = html.index('id="access-saved"')
    tag = html[saved_index:saved_index + 300]
    assert "hidden" in tag
    assert "Access code saved" in tag
    assert 'data-action="change-code"' in tag


def test_code_value_never_appears_in_the_page(monkeypatch):
    """
    The code is a server-side secret and should not be rendered. Even if a
    future change tries to pre-fill the field server-side, this stops it.
    """
    html = _page(monkeypatch)
    assert "judge-1" not in html
