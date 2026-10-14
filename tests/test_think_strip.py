from app.services.nebius_client import strip_think_blocks


def test_strip_closed_think_block():
    raw = "<think>I will think at length about this.</think>\nObserved: bar close."
    out = strip_think_blocks(raw)
    assert "think" not in out.lower()
    assert "Observed: bar close." in out


def test_strip_unclosed_think_prefix():
    raw = "<think>partial thoughts with no closing tag"
    out = strip_think_blocks(raw)
    assert out == ""  # nothing useful remains


def test_strip_mixed_content():
    raw = "Intro.\n<think>\nDeliberation\n</think>\n- setup: bar over mid-foot"
    out = strip_think_blocks(raw)
    assert "Deliberation" not in out
    assert "- setup: bar over mid-foot" in out
