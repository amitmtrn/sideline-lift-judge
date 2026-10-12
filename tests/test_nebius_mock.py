import asyncio
from unittest.mock import patch, Mock

from app.services.nebius_client import NebiusChatClient


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
    with patch("app.services.nebius_client.requests.post", return_value=m) as _post:
        text = asyncio.get_event_loop().run_until_complete(
            client.create_chat_completion(messages=[{"role": "user", "content": "hi"}], model=None)
        )
        assert '"lift":"snatch"' in text
        _post.assert_called_once()

