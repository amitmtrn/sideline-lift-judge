import asyncio
from unittest.mock import patch, Mock

from server.services.nebius_client import NebiusChatClient


def test_nebius_client_extracts_message_text():
    client = NebiusChatClient(api_key="test", base_url="https://example.test/v1", default_model="nvidia/Cosmos3-Super-Reasoner")
    fake_resp = {
        "id": "cmpl-123",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": '{"lift":"snatch","score":7,"band":"solid","max_horizontal_drift":"x","cues":["x"],"confidence":0.4}'},
                "finish_reason": "stop",
            }
        ],
    }
    m = Mock()
    m.status_code = 200
    m.json.return_value = fake_resp
    with patch("server.services.nebius_client.requests.post", return_value=m) as _post:
        text = asyncio.run(
            client.create_chat_completion(messages=[{"role": "user", "content": "hi"}], model=None)
        )
        assert '"lift":"snatch"' in text
        _post.assert_called_once()



def _reply(status, content='{"ok":true}'):
    m = Mock()
    m.status_code = status
    m.text = content
    m.json.return_value = {"choices": [{"message": {"content": content}}]}
    return m


def test_client_uses_a_connect_and_read_timeout(monkeypatch):
    from server.services import nebius_client

    monkeypatch.setenv("NEBIUS_TIMEOUT_SECONDS", "12")
    client = NebiusChatClient(api_key="k", base_url="https://example.test/v1")
    with patch("server.services.nebius_client.requests.post", return_value=_reply(200)) as post:
        asyncio.run(client.create_chat_completion(messages=[{"role": "user", "content": "hi"}], model="m"))
    assert post.call_args.kwargs["timeout"] == (nebius_client.CONNECT_TIMEOUT_SECONDS, 12.0)


def test_a_busy_provider_gets_exactly_one_more_try(monkeypatch):
    from server.services import nebius_client

    monkeypatch.setattr(nebius_client, "RETRY_BACKOFF_SECONDS", 0.0)
    client = NebiusChatClient(api_key="k", base_url="https://example.test/v1")
    with patch("server.services.nebius_client.requests.post", side_effect=[_reply(503, "down"), _reply(200)]) as post:
        text = asyncio.run(client.create_chat_completion(messages=[{"role": "user", "content": "hi"}], model="m"))
    assert text == '{"ok":true}' and post.call_count == 2

    with patch("server.services.nebius_client.requests.post", side_effect=[_reply(503, "down"), _reply(503, "down")]) as post:
        import pytest

        with pytest.raises(nebius_client.NebiusAPIError) as failure:
            asyncio.run(client.create_chat_completion(messages=[{"role": "user", "content": "hi"}], model="m"))
    assert failure.value.status == 503 and post.call_count == 2


def test_a_timeout_gets_exactly_one_more_try(monkeypatch):
    """A ReadTimeout from Token Factory is treated like a 503 — one retry."""
    import requests

    from server.services import nebius_client

    monkeypatch.setattr(nebius_client, "RETRY_BACKOFF_SECONDS", 0.0)
    client = NebiusChatClient(api_key="k", base_url="https://example.test/v1")

    # First attempt times out; second succeeds — the client should recover.
    attempt = {"n": 0}

    def side_effect(*args, **kwargs):  # noqa: ARG001
        attempt["n"] += 1
        if attempt["n"] == 1:
            raise requests.exceptions.ReadTimeout("provider read timeout")
        return _reply(200)

    with patch("server.services.nebius_client.requests.post", side_effect=side_effect) as post:
        text = asyncio.run(client.create_chat_completion(messages=[{"role": "user", "content": "hi"}], model="m"))
    assert text == '{"ok":true}' and post.call_count == 2

    # Two timeouts in a row surface the exception, so the fallback judge /
    # partial-result path upstream has something to react to.
    import pytest

    with patch(
        "server.services.nebius_client.requests.post",
        side_effect=requests.exceptions.ReadTimeout("provider read timeout"),
    ) as post:
        with pytest.raises(requests.exceptions.ReadTimeout):
            asyncio.run(client.create_chat_completion(messages=[{"role": "user", "content": "hi"}], model="m"))
    assert post.call_count == 2


def test_only_an_unknown_parameter_drops_the_no_thinking_hint(monkeypatch):
    from server.services import nebius_client

    monkeypatch.setattr(nebius_client, "RETRY_BACKOFF_SECONDS", 0.0)
    client = NebiusChatClient(api_key="k", base_url="https://example.test/v1")
    extra = nebius_client.NO_THINKING_EXTRA_BODY
    with patch("server.services.nebius_client.requests.post", side_effect=[_reply(400, "unknown field"), _reply(200)]) as post:
        asyncio.run(client.create_chat_completion(messages=[{"role": "user", "content": "hi"}], model="m", extra_body=extra))
    assert "chat_template_kwargs" in post.call_args_list[0].kwargs["json"]
    assert "chat_template_kwargs" not in post.call_args_list[1].kwargs["json"]
    # A 503 is not about the parameter: the retry keeps the hint.
    with patch("server.services.nebius_client.requests.post", side_effect=[_reply(503, "down"), _reply(200)]) as post:
        asyncio.run(client.create_chat_completion(messages=[{"role": "user", "content": "hi"}], model="m", extra_body=extra))
    assert all("chat_template_kwargs" in call.kwargs["json"] for call in post.call_args_list)


def test_judge_messages_carry_the_measurement_when_there_is_one():
    from server.services.nebius_client import assemble_judge_messages

    without = assemble_judge_messages(lift="snatch", observations="obs", prompts_dir="prompts")
    with_measurement = assemble_judge_messages(
        lift="snatch", observations="obs", prompts_dir="prompts", measurement="Bar-path measurement: 0.2 plate diameters"
    )
    assert len(with_measurement) == len(without) + 1
    assert with_measurement[-1]["role"] == "user"
    assert "0.2 plate diameters" in with_measurement[-1]["content"]
    guidance = with_measurement[1]["content"]
    assert "prefer the measurement" in guidance
    assert "prefer the measurement" not in without[1]["content"]
