from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from nanobot.cli.commands import app
from nanobot.config.schema import Config

runner = CliRunner()


def test_channels_login_wechat_success(monkeypatch, tmp_path: Path) -> None:
    config = Config()

    monkeypatch.setattr("nanobot.config.loader.load_config", lambda: config)
    monkeypatch.setattr("nanobot.config.loader.get_config_path", lambda: tmp_path / "config.json")

    async def _fake_login(self, timeout_seconds: int = 300) -> bool:  # type: ignore[no-untyped-def]
        self.config.ilink_bot_id = "bot-123"
        return True

    monkeypatch.setattr("nanobot.channels.wechat.WeChatChannel.login", _fake_login)

    result = runner.invoke(app, ["channels", "login", "--channel", "wechat"])
    assert result.exit_code == 0
    assert "WeChat login successful" in result.stdout
    assert "bot-123" in result.stdout


def test_channels_login_rejects_unknown_channel() -> None:
    result = runner.invoke(app, ["channels", "login", "--channel", "unknown"])
    assert result.exit_code == 1
    assert "Unsupported channel" in result.stdout
