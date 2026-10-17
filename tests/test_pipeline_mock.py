import base64
from unittest.mock import Mock, patch

import numpy as np  # type: ignore

from app.services.nebius_client import (
    NebiusChatClient,
    assemble_vision_messages,
    assemble_judge_messages,
    collect_observations_in_chunks,
    request_judgment,
)


def _single_image_data_url() -> str:
    import cv2  # type: ignore
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def test_two_stage_pipeline_mocks_requests():
    client = NebiusChatClient(api_key="x", base_url="https://example.test/v1")
    images = [_single_image_data_url()]
    v_msgs = assemble_vision_messages(lift="snatch", images=images, prompts_dir="prompts")
    j_msgs = assemble_judge_messages(lift="snatch", observations="Bar close; slight forward loop; catch stacked.", prompts_dir="prompts")

    # Mock two sequential responses: stage 1 (plain text), stage 2 (JSON)
    m1 = Mock()
    m1.status_code = 200
    m1.json.return_value = {"choices": [{"message": {"content": "Observed: bar stays close; minor drift at turnover; catch over mid-foot."}}]}
    m2 = Mock()
    m2.status_code = 200
    m2.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": '{"lift":"snatch","score":8,"band":"solid","max_horizontal_drift":"minor forward loop at turnover","cues":["keep the bar close to the body"],"confidence":0.7}'
                }
            }
        ]
    }
    import asyncio
    with patch("app.services.nebius_client.requests.post", side_effect=[m1, m2]) as post:
        loop = asyncio.get_event_loop()
        # Stage 1
        obs_text = loop.run_until_complete(client.create_chat_completion(messages=v_msgs, model="openbmb/MiniCPM-V-4_5"))
        assert "Observed" in obs_text
        # Stage 2
        out = loop.run_until_complete(client.create_chat_completion(messages=j_msgs, model="nvidia/Nemotron-3-Ultra-550b-a55b"))
        assert '"lift":"snatch"' in out
        assert post.call_count == 2


def test_assemble_accepts_lift_type_enum():
    # The web form binds lift as LiftType; str(LiftType.snatch) is "LiftType.snatch",
    # which used to be interpolated straight into the prompt filename.
    from app.schemas.judgment import LiftType

    j_msgs = assemble_judge_messages(lift=LiftType.snatch, observations="obs", prompts_dir="prompts")
    assert any("snatch" in str(m["content"]) for m in j_msgs)
    v_msgs = assemble_vision_messages(lift=LiftType.clean, images=[], prompts_dir="prompts")
    assert "Lift: clean" in v_msgs[1]["content"][0]["text"]


def test_judge_request_disables_thinking_and_raises_token_budget():
    client = NebiusChatClient(api_key="x", base_url="https://example.test/v1")
    m = Mock()
    m.status_code = 200
    m.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": '{"lift":"snatch","score":5,"band":"mixed","max_horizontal_drift":"forward loop",'
                    '"cues":["keep the bar close to the body"],"confidence":0.6}'
                }
            }
        ]
    }
    import asyncio
    with patch("app.services.nebius_client.requests.post", return_value=m) as post:
        judgment = asyncio.get_event_loop().run_until_complete(
            request_judgment(
                client=client,
                lift="snatch",
                observations="Bar drifts forward through the turnover.",
                prompts_dir="prompts",
                model="nvidia/Nemotron-3-Ultra-550b-a55b",
            )
        )
    assert judgment.score == 5
    payload = post.call_args.kwargs["json"]
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["max_tokens"] >= 2048


def test_judge_request_retries_once_when_reply_has_no_json():
    client = NebiusChatClient(api_key="x", base_url="https://example.test/v1")
    truncated = Mock()
    truncated.status_code = 200
    truncated.json.return_value = {"choices": [{"message": {"content": "Let me analyze the observations step by"}}]}
    good = Mock()
    good.status_code = 200
    good.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": '{"lift":"jerk","score":7,"band":"solid","max_horizontal_drift":"small dip drift",'
                    '"cues":["dip straight down (no forward knee slide); bar stays over mid-foot"],"confidence":0.7}'
                }
            }
        ]
    }
    import asyncio
    with patch("app.services.nebius_client.requests.post", side_effect=[truncated, good]) as post:
        judgment = asyncio.get_event_loop().run_until_complete(
            request_judgment(
                client=client,
                lift="jerk",
                observations="Dip is straight; bar stays close.",
                prompts_dir="prompts",
                model="nvidia/Nemotron-3-Ultra-550b-a55b",
            )
        )
    assert judgment.band == "solid"
    assert post.call_count == 2


def _good_judge_response(lift: str = "snatch") -> Mock:
    m = Mock()
    m.status_code = 200
    m.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": '{"lift":"%s","score":6,"band":"mixed","max_horizontal_drift":"forward loop in the second pull",'
                    '"cues":["keep the bar close to the body"],"confidence":0.5}' % lift
                }
            }
        ]
    }
    return m


def _api_error_response() -> Mock:
    m = Mock()
    m.status_code = 503
    m.text = "upstream unavailable"
    return m


def test_judge_falls_back_to_second_model_when_primary_fails():
    # AC-7 / FR-6: Ultra errors out, Super answers. A failing call costs two posts
    # because the client retries once without the no-thinking hint.
    client = NebiusChatClient(api_key="x", base_url="https://example.test/v1")
    import asyncio
    with patch(
        "app.services.nebius_client.requests.post",
        side_effect=[_api_error_response(), _api_error_response(), _good_judge_response()],
    ) as post:
        judgment = asyncio.get_event_loop().run_until_complete(
            request_judgment(
                client=client,
                lift="snatch",
                observations="Bar loops forward off the floor.",
                prompts_dir="prompts",
                model="nvidia/Nemotron-3-Ultra-550b-a55b",
                fallback_model="nvidia/nemotron-3-super-120b-a12b",
            )
        )
    assert judgment.score == 6
    assert post.call_count == 3
    models = [call.kwargs["json"]["model"] for call in post.call_args_list]
    assert models[:2] == ["nvidia/Nemotron-3-Ultra-550b-a55b"] * 2
    assert models[2] == "nvidia/nemotron-3-super-120b-a12b"


def test_judge_falls_back_when_primary_never_returns_json():
    client = NebiusChatClient(api_key="x", base_url="https://example.test/v1")
    prose = Mock()
    prose.status_code = 200
    prose.json.return_value = {"choices": [{"message": {"content": "Let me think about the observations"}}]}
    import asyncio
    with patch(
        "app.services.nebius_client.requests.post",
        side_effect=[prose, prose, _good_judge_response("clean")],
    ) as post:
        judgment = asyncio.get_event_loop().run_until_complete(
            request_judgment(
                client=client,
                lift="clean",
                observations="Bar close through the first pull.",
                prompts_dir="prompts",
                model="nvidia/Nemotron-3-Ultra-550b-a55b",
                fallback_model="nvidia/nemotron-3-super-120b-a12b",
            )
        )
    # Primary: first reply + repair retry, both prose. Fallback: valid JSON.
    assert judgment.lift.value == "clean"
    assert post.call_count == 3
    assert post.call_args_list[2].kwargs["json"]["model"] == "nvidia/nemotron-3-super-120b-a12b"


def test_judge_raises_primary_error_when_fallback_disabled():
    client = NebiusChatClient(api_key="x", base_url="https://example.test/v1")
    import asyncio
    import pytest

    with patch(
        "app.services.nebius_client.requests.post",
        side_effect=[_api_error_response(), _api_error_response()],
    ) as post:
        with pytest.raises(RuntimeError):
            asyncio.get_event_loop().run_until_complete(
                request_judgment(
                    client=client,
                    lift="snatch",
                    observations="obs",
                    prompts_dir="prompts",
                    model="nvidia/Nemotron-3-Ultra-550b-a55b",
                    fallback_model="",
                )
            )
    assert post.call_count == 2


def test_judge_fallback_model_env_override(monkeypatch):
    from app.services.nebius_client import DEFAULT_JUDGE_FALLBACK_MODEL, judge_fallback_model

    monkeypatch.delenv("NEBIUS_JUDGE_FALLBACK_MODEL", raising=False)
    assert judge_fallback_model() == DEFAULT_JUDGE_FALLBACK_MODEL
    monkeypatch.setenv("NEBIUS_JUDGE_FALLBACK_MODEL", "nvidia/other-judge")
    assert judge_fallback_model() == "nvidia/other-judge"
    # Empty value is the documented way to turn the fallback off.
    monkeypatch.setenv("NEBIUS_JUDGE_FALLBACK_MODEL", "")
    assert judge_fallback_model() == ""


def test_vision_chunking_makes_multiple_calls_when_over_limit():
    client = NebiusChatClient(api_key="x", base_url="https://example.test/v1")
    # Build 12 tiny images → should split into 2 calls when max_images=10
    images = [_single_image_data_url() for _ in range(12)]
    m1 = Mock(); m1.status_code = 200; m1.json.return_value = {"choices": [{"message": {"content": "obs1"}}]}
    m2 = Mock(); m2.status_code = 200; m2.json.return_value = {"choices": [{"message": {"content": "obs2"}}]}
    with patch("app.services.nebius_client.requests.post", side_effect=[m1, m2]) as post:
        import asyncio
        text = asyncio.get_event_loop().run_until_complete(
            collect_observations_in_chunks(
                client=client,
                lift="snatch",
                images=images,
                prompts_dir="prompts",
                model="openbmb/MiniCPM-V-4_5",
                max_images_per_call=10,
            )
        )
        assert "obs1" in text and "obs2" in text
        assert post.call_count == 2
