"""WeChat channel implementation using ilink long-poll APIs."""

from __future__ import annotations

import asyncio
import base64
import json
import random
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any

import httpx
from loguru import logger
from pydantic import Field

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.paths import get_runtime_subdir
from nanobot.config.schema import Base

DEFAULT_BOT_TYPE = "3"
DEFAULT_API_BASE_URL = "https://ilinkai.weixin.qq.com"
DEFAULT_POLL_TIMEOUT_MS = 35_000
DEFAULT_REQUEST_TIMEOUT_SECONDS = 15.0
SESSION_EXPIRED_ERRCODE = -14
MAX_CONTEXT_TOKEN_CACHE = 5_000
CHANNEL_VERSION = "nanobot-wechat/0.1.0"


class WeChatConfig(Base):
    """WeChat channel configuration."""

    enabled: bool = False
    bot_type: str = DEFAULT_BOT_TYPE
    api_base_url: str = DEFAULT_API_BASE_URL
    allow_from: list[str] = Field(default_factory=list)
    bot_token: str = ""
    ilink_bot_id: str = ""
    poll_timeout_ms: int = DEFAULT_POLL_TIMEOUT_MS
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS


class WeChatChannel(BaseChannel):
    """WeChat channel using ilink getUpdates/sendMessage APIs."""

    name = "wechat"
    display_name = "WeChat"

    ENDPOINTS = {
        "get_bot_qrcode": "ilink/bot/get_bot_qrcode",
        "get_qrcode_status": "ilink/bot/get_qrcode_status",
        "getupdates": "ilink/bot/getupdates",
        "sendmessage": "ilink/bot/sendmessage",
    }

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return WeChatConfig().model_dump(by_alias=True)

    def __init__(self, config: Any, bus: MessageBus, state_dir: Path | None = None):
        if isinstance(config, dict):
            config = WeChatConfig.model_validate(config)
        super().__init__(config, bus)
        self.config: WeChatConfig = config
        self.state_dir = (state_dir or get_runtime_subdir("wechat")).expanduser()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.credentials_path = self.state_dir / "credentials.json"
        self.sync_buf_path = self.state_dir / "sync_buf.json"
        self._client: httpx.AsyncClient | None = None
        self._poll_task: asyncio.Task[None] | None = None
        self._get_updates_buf = ""
        self._context_tokens: OrderedDict[str, str] = OrderedDict()
        self._load_sync_buf()
        self.load_credentials()

    async def start(self) -> None:
        """Start WeChat long-polling monitor."""
        if not self.config.bot_token:
            logger.error("WeChat bot token is not configured. Run: nanobot channels login --channel wechat")
            return

        self._running = True
        self._client = httpx.AsyncClient(
            base_url=self.config.api_base_url,
            timeout=httpx.Timeout(max(float(self.config.request_timeout_seconds), 5.0)),
        )
        self._poll_task = asyncio.create_task(self._poll_updates_loop())

        logger.info("Starting WeChat channel (long-poll)")
        while self._running:
            await asyncio.sleep(1)

    async def stop(self) -> None:
        """Stop WeChat polling and close HTTP client."""
        self._running = False

        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None

        if self._client:
            await self._client.aclose()
            self._client = None

        logger.info("WeChat channel stopped")

    async def send(self, msg: OutboundMessage) -> None:
        """Send text reply via WeChat sendmessage."""
        if not self._client:
            logger.warning("WeChat client is not running")
            return
        if not self.config.bot_token:
            logger.warning("WeChat bot token missing, cannot send")
            return

        context_token = ""
        if isinstance(msg.metadata, dict):
            context_token = str(msg.metadata.get("context_token") or "").strip()
        if not context_token:
            context_token = self._context_tokens.get(msg.chat_id, "").strip()
        if not context_token:
            logger.warning("Missing context_token for WeChat user {}, skipping outbound", msg.chat_id)
            return

        if msg.media:
            logger.warning("WeChat media sending is not implemented yet; sending text only")

        text = (msg.content or "").strip()
        if not text and msg.media:
            text = "[attachment omitted]"
        if not text:
            logger.debug("Skipping empty WeChat outbound message")
            return

        body = {
            "msg": {
                "from_user_id": "",
                "to_user_id": msg.chat_id,
                "client_id": self._generate_client_id(),
                "message_type": 2,  # BOT
                "message_state": 2,  # FINISH
                "context_token": context_token,
                "item_list": [
                    {
                        "type": 1,  # TEXT
                        "text_item": {"text": text},
                    }
                ],
            },
            "base_info": self._base_info(),
        }
        await self._post_logged_json(
            endpoint=self.ENDPOINTS["sendmessage"],
            body=body,
            timeout=max(float(self.config.request_timeout_seconds), 5.0),
        )

    async def login(self, timeout_seconds: int = 300) -> bool:
        """Login with QR code and persist bot credentials."""
        timeout_seconds = max(30, int(timeout_seconds))
        timeout = max(float(self.config.request_timeout_seconds), 5.0)
        async with httpx.AsyncClient(base_url=self.config.api_base_url, timeout=httpx.Timeout(timeout)) as client:
            try:
                resp = await client.get(
                    self.ENDPOINTS["get_bot_qrcode"],
                    params={"bot_type": self.config.bot_type or DEFAULT_BOT_TYPE},
                )
                resp.raise_for_status()
                qr_data = resp.json()
            except Exception as exc:
                logger.error("Failed to fetch WeChat QR code: {}", exc)
                return False

            if not self._is_success_response(qr_data):
                logger.error("WeChat QR request failed: {}", qr_data)
                return False

            qrcode = str(qr_data.get("qrcode") or "").strip()
            qr_url = str(qr_data.get("qrcode_img_content") or "").strip()
            if not qrcode:
                logger.error("WeChat QR response missing qrcode")
                return False

            self._display_qr(qr_url or qrcode)
            logger.info("Waiting for WeChat QR confirmation...")
            deadline = time.monotonic() + timeout_seconds
            scanned_notice_shown = False

            while time.monotonic() < deadline:
                try:
                    poll_timeout = max(float(self.config.poll_timeout_ms) / 1000.0 + 5.0, 10.0)
                    poll_resp = await client.get(
                        self.ENDPOINTS["get_qrcode_status"],
                        params={"qrcode": qrcode},
                        headers={"iLink-App-ClientVersion": "1"},
                        timeout=poll_timeout,
                    )
                    poll_resp.raise_for_status()
                    status_data = poll_resp.json()
                except httpx.TimeoutException:
                    continue
                except Exception as exc:
                    logger.warning("Error polling WeChat QR status: {}", exc)
                    await asyncio.sleep(1)
                    continue

                status = str(status_data.get("status") or "").strip().lower()
                if status == "wait":
                    continue
                if status == "scaned":
                    if not scanned_notice_shown:
                        logger.info("QR scanned, waiting for confirmation in WeChat")
                        scanned_notice_shown = True
                    continue
                if status == "expired":
                    logger.error("WeChat QR expired. Please retry login.")
                    return False
                if status == "confirmed":
                    bot_token = str(status_data.get("bot_token") or "").strip()
                    ilink_bot_id = str(status_data.get("ilink_bot_id") or "").strip()
                    if not bot_token or not ilink_bot_id:
                        logger.error("WeChat login confirmed but missing bot_token/ilink_bot_id")
                        return False
                    self.config.bot_token = bot_token
                    self.config.ilink_bot_id = ilink_bot_id
                    self._save_credentials()
                    logger.info("WeChat login successful (bot_id={})", ilink_bot_id)
                    return True

                logger.debug("Unhandled WeChat QR status payload: {}", status_data)

        logger.error("WeChat login timed out")
        return False

    def load_credentials(self) -> bool:
        """Load persisted credentials if config has no token."""
        if self.config.bot_token and self.config.ilink_bot_id:
            return True
        if not self.credentials_path.exists():
            return False
        try:
            data = json.loads(self.credentials_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to load WeChat credentials: {}", exc)
            return False

        self.config.bot_token = str(data.get("bot_token") or self.config.bot_token or "").strip()
        self.config.ilink_bot_id = str(data.get("ilink_bot_id") or self.config.ilink_bot_id or "").strip()
        saved_buf = str(data.get("get_updates_buf") or "").strip()
        if saved_buf and not self._get_updates_buf:
            self._get_updates_buf = saved_buf
        return bool(self.config.bot_token and self.config.ilink_bot_id)

    async def _poll_updates_loop(self) -> None:
        """Long-poll loop for getupdates."""
        logger.info("WeChat polling loop started")
        consecutive_errors = 0

        while self._running:
            try:
                poll_timeout = max(float(self.config.poll_timeout_ms) / 1000.0 + 5.0, 10.0)
                payload = {
                    "get_updates_buf": self._get_updates_buf,
                    "base_info": self._base_info(),
                }
                data = await self._post_logged_json(
                    endpoint=self.ENDPOINTS["getupdates"],
                    body=payload,
                    timeout=poll_timeout,
                )
                consecutive_errors = 0

                new_buf = str(data.get("get_updates_buf") or "").strip()
                if new_buf and new_buf != self._get_updates_buf:
                    self._get_updates_buf = new_buf
                    self._save_sync_buf()

                msgs = data.get("msgs") or []
                if not isinstance(msgs, list):
                    continue
                for raw_msg in msgs:
                    if isinstance(raw_msg, dict):
                        await self._handle_incoming_message(raw_msg)
            except asyncio.CancelledError:
                break
            except httpx.TimeoutException:
                # Long-poll timeout without messages is expected.
                continue
            except Exception as exc:
                message = str(exc)
                if f"errcode={SESSION_EXPIRED_ERRCODE}" in message:
                    logger.error("WeChat session expired. Run: nanobot channels login --channel wechat")
                    self._running = False
                    break

                consecutive_errors += 1
                backoff = min(30.0, max(1.0, float(2 ** min(consecutive_errors, 4))))
                logger.warning("WeChat polling error: {} (retry in {}s)", exc, backoff)
                await asyncio.sleep(backoff)

        logger.info("WeChat polling loop stopped")

    async def _handle_incoming_message(self, msg: dict[str, Any]) -> None:
        """Normalize WeChat message payload and forward to bus."""
        try:
            from_user_id = str(msg.get("from_user_id") or "").strip()
            if not from_user_id:
                return

            context_token = str(msg.get("context_token") or "").strip()
            if context_token:
                self._remember_context_token(from_user_id, context_token)

            item_list = msg.get("item_list")
            content = self._extract_content(item_list if isinstance(item_list, list) else [])
            if not content:
                # Compatibility fallback for legacy payloads.
                content = str(msg.get("text") or "").strip()
            if not content:
                return

            metadata = {
                "context_token": context_token,
                "message_id": str(msg.get("message_id") or ""),
                "message_type": str(msg.get("message_type") or ""),
            }
            await self._handle_message(
                sender_id=from_user_id,
                chat_id=from_user_id,
                content=content,
                metadata=metadata,
            )
        except Exception as exc:
            logger.error("Error handling WeChat inbound message: {}", exc)

    def _save_credentials(self) -> None:
        payload = {
            "bot_token": self.config.bot_token,
            "ilink_bot_id": self.config.ilink_bot_id,
            "bot_type": self.config.bot_type,
            "saved_at": int(time.time()),
            "get_updates_buf": self._get_updates_buf,
        }
        self.credentials_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        try:
            self.credentials_path.chmod(0o600)
        except Exception:
            pass

    def _load_sync_buf(self) -> None:
        if not self.sync_buf_path.exists():
            return
        try:
            data = json.loads(self.sync_buf_path.read_text(encoding="utf-8"))
        except Exception:
            return
        self._get_updates_buf = str(data.get("get_updates_buf") or "").strip()

    def _save_sync_buf(self) -> None:
        payload = {"get_updates_buf": self._get_updates_buf}
        self.sync_buf_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        try:
            self.sync_buf_path.chmod(0o600)
        except Exception:
            pass

    def _remember_context_token(self, user_id: str, context_token: str) -> None:
        if user_id in self._context_tokens:
            self._context_tokens.move_to_end(user_id)
        self._context_tokens[user_id] = context_token
        while len(self._context_tokens) > MAX_CONTEXT_TOKEN_CACHE:
            self._context_tokens.popitem(last=False)

    def _build_auth_headers(self) -> dict[str, str]:
        uin = random.randint(0, 2**32 - 1)
        uin_b64 = base64.b64encode(str(uin).encode("utf-8")).decode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "AuthorizationType": "ilink_bot_token",
            "X-WECHAT-UIN": uin_b64,
        }
        token = (self.config.bot_token or "").strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    @staticmethod
    def _is_success_response(data: dict[str, Any]) -> bool:
        ret = data.get("ret")
        if ret is not None and int(ret) != 0:
            return False
        errcode = data.get("errcode")
        if ret is None and errcode is not None and int(errcode) != 0:
            return False
        return True

    async def _post_logged_json(
        self,
        *,
        endpoint: str,
        body: dict[str, Any],
        timeout: float,
    ) -> dict[str, Any]:
        if not self._client:
            raise RuntimeError("WeChat client is not initialized")

        response = await self._client.post(
            endpoint,
            json=body,
            headers=self._build_auth_headers(),
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise RuntimeError(f"Unexpected WeChat API response type: {type(data)}")
        if not self._is_success_response(data):
            errcode = data.get("errcode", data.get("ret", "unknown"))
            errmsg = data.get("errmsg", "unknown error")
            raise RuntimeError(f"WeChat API error: errcode={errcode} errmsg={errmsg}")
        return data

    @staticmethod
    def _extract_content(item_list: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for item in item_list:
            if not isinstance(item, dict):
                continue
            item_type = int(item.get("type") or 0)
            if item_type == 1:
                text = str((item.get("text_item") or {}).get("text") or "").strip()
                if text:
                    parts.append(text)
            elif item_type == 2:
                parts.append("[image]")
            elif item_type == 3:
                voice_text = str((item.get("voice_item") or {}).get("text") or "").strip()
                parts.append(voice_text or "[voice]")
            elif item_type == 4:
                file_name = str((item.get("file_item") or {}).get("file_name") or "").strip()
                parts.append(f"[file: {file_name}]" if file_name else "[file]")
            elif item_type == 5:
                parts.append("[video]")
        return "\n".join(parts).strip()

    @staticmethod
    def _base_info() -> dict[str, Any]:
        return {"channel_version": CHANNEL_VERSION}

    @staticmethod
    def _generate_client_id() -> str:
        return f"nanobot-wechat-{uuid.uuid4().hex}"

    @staticmethod
    def _display_qr(content: str) -> None:
        print("\nScan the QR code with WeChat and confirm login:\n")
        if content.startswith("http://") or content.startswith("https://"):
            print(f"QR URL: {content}\n")
        try:
            import qrcode

            qr = qrcode.QRCode(border=1)
            qr.add_data(content)
            qr.print_ascii(invert=True)
            print()
        except Exception:
            print(content)
            print()
