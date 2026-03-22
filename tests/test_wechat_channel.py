from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from nanobot.bus.events import OutboundMessage
from nanobot.channels.wechat import WeChatChannel


class _DummyBus:
    def __init__(self) -> None:
        self.inbound: list[Any] = []

    async def publish_inbound(self, msg: Any) -> None:
        self.inbound.append(msg)


@pytest.mark.asyncio
async def test_wechat_inbound_text_item_list(tmp_path: Path) -> None:
    bus = _DummyBus()
    channel = WeChatChannel(
        {
            "enabled": True,
            "allowFrom": ["*"],
            "botToken": "tok",
            "ilinkBotId": "bot",
        },
        bus,  # type: ignore[arg-type]
        state_dir=tmp_path / "wechat",
    )

    await channel._handle_incoming_message(
        {
            "from_user_id": "user-1",
            "message_id": 123,
            "context_token": "ctx-1",
            "item_list": [
                {"type": 1, "text_item": {"text": "hello from item_list"}},
            ],
        }
    )

    assert len(bus.inbound) == 1
    msg = bus.inbound[0]
    assert msg.sender_id == "user-1"
    assert msg.chat_id == "user-1"
    assert msg.content == "hello from item_list"
    assert msg.metadata["context_token"] == "ctx-1"
    assert channel._context_tokens["user-1"] == "ctx-1"


@pytest.mark.asyncio
async def test_wechat_inbound_media_placeholder_not_dropped(tmp_path: Path) -> None:
    bus = _DummyBus()
    channel = WeChatChannel(
        {
            "enabled": True,
            "allowFrom": ["*"],
            "botToken": "tok",
            "ilinkBotId": "bot",
        },
        bus,  # type: ignore[arg-type]
        state_dir=tmp_path / "wechat",
    )

    await channel._handle_incoming_message(
        {
            "from_user_id": "user-2",
            "context_token": "ctx-2",
            "item_list": [
                {"type": 2, "image_item": {"media": {"encrypt_query_param": "abc"}}},
            ],
        }
    )

    assert len(bus.inbound) == 1
    assert bus.inbound[0].content == "[image]"


@pytest.mark.asyncio
async def test_wechat_send_prefers_metadata_context_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bus = _DummyBus()
    channel = WeChatChannel(
        {
            "enabled": True,
            "allowFrom": ["*"],
            "botToken": "tok",
            "ilinkBotId": "bot",
        },
        bus,  # type: ignore[arg-type]
        state_dir=tmp_path / "wechat",
    )
    channel._client = object()  # bypass running client guard

    captured: dict[str, Any] = {}

    async def _fake_post_logged_json(*, endpoint: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
        captured["endpoint"] = endpoint
        captured["body"] = body
        captured["timeout"] = timeout
        return {"ret": 0}

    monkeypatch.setattr(channel, "_post_logged_json", _fake_post_logged_json)

    await channel.send(
        OutboundMessage(
            channel="wechat",
            chat_id="user-3",
            content="hello outbound",
            metadata={"context_token": "ctx-meta"},
        )
    )

    assert captured["endpoint"] == WeChatChannel.ENDPOINTS["sendmessage"]
    assert captured["body"]["msg"]["context_token"] == "ctx-meta"
    assert captured["body"]["msg"]["to_user_id"] == "user-3"


@pytest.mark.asyncio
async def test_wechat_send_falls_back_to_cached_context_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bus = _DummyBus()
    channel = WeChatChannel(
        {
            "enabled": True,
            "allowFrom": ["*"],
            "botToken": "tok",
            "ilinkBotId": "bot",
        },
        bus,  # type: ignore[arg-type]
        state_dir=tmp_path / "wechat",
    )
    channel._client = object()  # bypass running client guard
    channel._context_tokens["user-4"] = "ctx-cache"

    captured: dict[str, Any] = {}

    async def _fake_post_logged_json(*, endpoint: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
        captured["body"] = body
        return {"ret": 0}

    monkeypatch.setattr(channel, "_post_logged_json", _fake_post_logged_json)

    await channel.send(OutboundMessage(channel="wechat", chat_id="user-4", content="cache send"))
    assert captured["body"]["msg"]["context_token"] == "ctx-cache"


@pytest.mark.asyncio
async def test_wechat_send_skips_when_context_token_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bus = _DummyBus()
    channel = WeChatChannel(
        {
            "enabled": True,
            "allowFrom": ["*"],
            "botToken": "tok",
            "ilinkBotId": "bot",
        },
        bus,  # type: ignore[arg-type]
        state_dir=tmp_path / "wechat",
    )
    channel._client = object()  # bypass running client guard

    called = {"value": False}

    async def _fake_post_logged_json(*, endpoint: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
        called["value"] = True
        return {"ret": 0}

    monkeypatch.setattr(channel, "_post_logged_json", _fake_post_logged_json)

    await channel.send(OutboundMessage(channel="wechat", chat_id="unknown-user", content="no ctx"))
    assert called["value"] is False


def test_wechat_success_response_allows_missing_ret() -> None:
    assert WeChatChannel._is_success_response({"qrcode": "abc"})
    assert WeChatChannel._is_success_response({"ret": 0, "msgs": []})
    assert not WeChatChannel._is_success_response({"ret": -1, "errmsg": "failed"})
    assert not WeChatChannel._is_success_response({"errcode": -14, "errmsg": "expired"})
