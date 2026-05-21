from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import requests

from daily_macro.nowcast_storage import NowcastStorage
from daily_macro.nowcasting import refresh_nowcasts


class _Http500Response:
    status_code = 500


class NowcastingTests(unittest.TestCase):
    def test_refresh_nowcasts_marks_transient_gdpnow_http_errors_partial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            http_error = requests.HTTPError("HTTP 500", response=_Http500Response())

            with patch(
                "daily_macro.nowcasting.fetch_cleveland_fed_nowcasts",
                return_value=[
                    {
                        "series_id": "cleveland_cpi_mom",
                        "target_period": "2026-04",
                        "value": 0.3,
                        "as_of_date": "2026-04-18",
                    }
                ],
            ), patch(
                "daily_macro.nowcasting.fetch_gdpnow_from_fred",
                side_effect=http_error,
            ), patch.dict(
                "os.environ",
                {"FRED_API_KEY": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},
                clear=False,
            ):
                result = refresh_nowcasts(data_dir=str(data_dir))

            self.assertEqual(result["cleveland"]["status"], "success")
            self.assertEqual(result["cleveland"]["count"], 3)
            self.assertEqual(result["gdpnow"]["status"], "partial_failure")
            self.assertEqual(result["gdpnow"]["count"], 0)
            self.assertEqual(result["gdpnow"]["error"], "HTTP 500")

            storage = NowcastStorage(data_dir / "macro.sqlite")
            try:
                latest = storage.fetch_latest("cleveland_cpi_mom")
            finally:
                storage.close()

            self.assertIsNotNone(latest)
            assert latest is not None
            self.assertEqual(latest["target_period"], "2026-04")

    def test_refresh_nowcasts_rejects_malformed_fred_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)

            with patch(
                "daily_macro.nowcasting.fetch_cleveland_fed_nowcasts",
                return_value=[],
            ), patch.dict(
                "os.environ",
                {"FRED_API_KEY": "bad-key"},
                clear=False,
            ):
                result = refresh_nowcasts(data_dir=str(data_dir))

            self.assertEqual(result["gdpnow"]["status"], "failed")
            self.assertEqual(
                result["gdpnow"]["error"],
                "Invalid FRED_API_KEY format. Expected a 32 character lower-case alpha-numeric string.",
            )
