from __future__ import annotations

from unittest.mock import MagicMock, patch

from daily_market.notifier import (
    _build_subject,
    _get_gmail_credentials,
    send_market_summary,
)


def _dummy_summary(status: str = "success") -> dict:
    return {
        "plain": "Test plain body",
        "html": "<p>Test HTML body</p>",
        "status": status,
        "ticker_count": 5,
        "success_count": 5,
    }


def test_build_subject_success():
    subj = _build_subject(_dummy_summary("success"), "morning", "2026-04-05")
    assert "[DAILY MARKET]" in subj
    assert "Morning Brief" in subj
    assert "2026-04-05" in subj
    assert "[SUCCESS]" not in subj  # no status suffix when success


def test_build_subject_partial():
    subj = _build_subject(_dummy_summary("partial"), "evening", "2026-04-05")
    assert "[PARTIAL]" in subj
    assert "Evening Brief" in subj


def test_send_email_missing_credentials():
    with patch.dict("os.environ", {}, clear=True):
        with patch("daily_market.notifier._load_local_config", return_value={}):
            try:
                send_market_summary(_dummy_summary(), session="morning", date_str="2026-04-05")
                assert False, "Expected RuntimeError"
            except RuntimeError as exc:
                assert "credentials" in str(exc).lower()


def test_send_email_calls_smtp():
    with patch.dict("os.environ", {
        "GMAIL_SENDER": "sender@example.com",
        "GMAIL_APP_PASSWORD": "testpw",
        "GMAIL_RECIPIENT": "recipient@example.com",
    }):
        with patch("daily_market.notifier.smtplib.SMTP_SSL") as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__ = MagicMock(return_value=mock_server)
            mock_smtp.return_value.__exit__ = MagicMock(return_value=False)
            ok, msg = send_market_summary(_dummy_summary(), session="morning", date_str="2026-04-05")

        assert ok is True
        assert "recipient@example.com" in msg
        mock_server.login.assert_called_once_with("sender@example.com", "testpw")
        mock_server.sendmail.assert_called_once()


def test_get_credentials_from_env():
    with patch.dict("os.environ", {
        "GMAIL_SENDER": "s@x.com",
        "GMAIL_APP_PASSWORD": "pw",
        "GMAIL_RECIPIENT": "r@x.com",
    }):
        sender, pw, recipient = _get_gmail_credentials()
        assert sender == "s@x.com"
        assert pw == "pw"
        assert recipient == "r@x.com"
