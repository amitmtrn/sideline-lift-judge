import base64
from unittest.mock import Mock, patch

import numpy as np  # type: ignore

from app.services.nebius_client import (
    NebiusChatClient,
    assemble_vision_messages,
    assemble_judge_messages,
    collect_observations_in_chunks,
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
