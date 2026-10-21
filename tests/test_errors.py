"""Failures reach the phone as one plain sentence, never as the provider's text."""
import asyncio

import requests

from server.services import errors
from server.services.nebius_client import NebiusAPIError


def test_provider_errors_map_to_busy_or_misconfigured():
    assert errors.user_message(NebiusAPIError(503, "<html>upstream</html>")) == errors.BUSY
    assert errors.user_message(NebiusAPIError(429, "slow down")) == errors.BUSY
    assert errors.user_message(NebiusAPIError(401, "bad key")) == errors.MISCONFIGURED
    assert errors.user_message(RuntimeError("Nebius API error 502: gateway")) == errors.BUSY
    assert errors.user_message(RuntimeError("Nebius API error 400: bad request")) == errors.GENERIC


def test_network_trouble_is_named_as_such():
    assert errors.user_message(requests.exceptions.ReadTimeout()) == errors.SLOW
    assert errors.user_message(requests.exceptions.ConnectionError()) == errors.UNREACHABLE
    assert errors.user_message(asyncio.TimeoutError()) == errors.TOO_LONG


def test_bad_json_from_the_judge_is_a_verdict_problem():
    assert errors.user_message(ValueError("no JSON object found")) == errors.NO_VERDICT


def test_anything_else_is_generic_and_never_leaks_the_text():
    message = errors.user_message(RuntimeError("Traceback: secret internals"))
    assert message == errors.GENERIC
    assert "secret" not in message


def test_detail_keeps_the_raw_error_for_logs():
    detail = errors.detail_of(RuntimeError("Nebius API error 503: upstream"))
    assert detail.startswith("RuntimeError: Nebius API error 503")
