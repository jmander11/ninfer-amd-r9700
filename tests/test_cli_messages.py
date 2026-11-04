"""Messages files retain their local media meaning outside their containing directory."""

import json
from pathlib import Path

import pytest

from examples.cli.send_to_serve import load_cli_messages
from tools.reference.qwen3.common.multimodal import load_messages


@pytest.mark.parametrize("reader", [load_cli_messages, load_messages])
def test_file_relative_media(tmp_path: Path, reader):
    content = [
        {"type": "image", "image": "media/chart.png"},
        {"type": "video", "video": "file://media/video.mp4"},
        {"type": "image_url", "image_url": {"url": "media/photo.png"}},
        {"type": "image_url", "image_url": {"url": "https://example.invalid/image.png"}},
        {"type": "image", "image": "data:image/png;base64,AA=="},
        {"type": "text", "text": "media/this is literal text"},
    ]
    messages_file = tmp_path / "messages.json"
    messages_file.write_text(json.dumps([{"role": "user", "content": content}]), encoding="utf-8")
    assert Path.cwd() != tmp_path
    loaded = reader(messages_file)
    messages = loaded[0] if isinstance(loaded, tuple) else loaded
    actual = messages[0]["content"]
    assert actual[0]["image"] == str(tmp_path / "media/chart.png")
    assert actual[1]["video"] == str(tmp_path / "media/video.mp4")
    assert actual[2]["image_url"]["url"] == str(tmp_path / "media/photo.png")
    assert actual[3:] == content[3:]
