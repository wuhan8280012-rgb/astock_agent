"""Tests for data_pipeline/tushare_client.py — using mocked Tushare API."""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
import time

from data_pipeline.tushare_client import TushareClient, TushareRequestError


@pytest.fixture
def mock_client():
    """Create a TushareClient with mocked tushare API."""
    with patch("data_pipeline.tushare_client.ts") as mock_ts:
        mock_api = MagicMock()
        mock_ts.pro_api.return_value = mock_api
        client = TushareClient(token="test_token", rate_limit=600)  # High limit to avoid waits in tests
        client._api = mock_api
        yield client, mock_api


class TestRateLimiting:
    def test_rate_limit_enforced(self, mock_client):
        """Verify that calls respect rate limiting."""
        client, mock_api = mock_client
        client._rate_limit = 60  # 1 call per second
        client._min_interval = 1.0

        mock_api.daily.return_value = pd.DataFrame({"ts_code": ["000001.SZ"], "close": [10.0]})

        start = time.time()
        client.query("daily", ts_code="000001.SZ")
        client.query("daily", ts_code="000001.SZ")
        elapsed = time.time() - start

        # Second call should wait ~1 second
        assert elapsed >= 0.9


class TestRetry:
    def test_retry_then_succeed(self, mock_client):
        """First call fails, second succeeds."""
        client, mock_api = mock_client
        good_df = pd.DataFrame({"ts_code": ["000001.SZ"], "close": [10.0]})
        mock_api.daily.side_effect = [Exception("timeout"), good_df]

        result = client.query("daily", ts_code="000001.SZ", max_retries=2, retry_interval=0)
        assert len(result) == 1

    def test_all_retries_exhausted(self, mock_client):
        """All retries fail — raise TushareRequestError."""
        client, mock_api = mock_client
        mock_api.daily.side_effect = Exception("persistent error")

        with pytest.raises(TushareRequestError, match="All 3 retries failed"):
            client.query("daily", ts_code="000001.SZ", max_retries=3, retry_interval=0)


class TestDataValidation:
    def test_null_price_rows_dropped(self, mock_client):
        """Rows with null price fields are dropped."""
        client, mock_api = mock_client
        df = pd.DataFrame({
            "ts_code": ["000001.SZ", "000002.SZ"],
            "close": [10.0, None],
            "open": [9.5, None],
            "high": [10.5, None],
            "low": [9.0, None],
        })
        mock_api.daily.return_value = df

        result = client.query("daily", ts_code="000001.SZ")
        assert len(result) == 1

    def test_duplicate_rows_deduped(self, mock_client):
        """Duplicate (ts_code, trade_date) rows keep the last one."""
        client, mock_api = mock_client
        df = pd.DataFrame({
            "ts_code": ["000001.SZ", "000001.SZ"],
            "trade_date": ["20240103", "20240103"],
            "close": [10.0, 10.5],
        })
        mock_api.daily.return_value = df

        result = client.query("daily", ts_code="000001.SZ")
        assert len(result) == 1
        assert result.iloc[0]["close"] == 10.5

    def test_outlier_warning_not_dropped(self, mock_client):
        """Rows with |pct_chg| > 22% trigger warning but are NOT dropped."""
        client, mock_api = mock_client
        df = pd.DataFrame({
            "ts_code": ["000001.SZ", "000002.SZ"],
            "trade_date": ["20240103", "20240103"],
            "pct_chg": [5.0, 25.0],
            "close": [10.0, 15.0],
        })
        mock_api.daily.return_value = df

        result = client.query("daily")
        assert len(result) == 2  # Outlier not dropped

    def test_empty_dataframe(self, mock_client):
        """Empty DataFrame returns without error."""
        client, mock_api = mock_client
        mock_api.daily.return_value = pd.DataFrame()

        result = client.query("daily", ts_code="000001.SZ")
        assert result.empty


class TestTradeCalendar:
    def test_get_trade_calendar(self, mock_client):
        """Trade calendar returned and cached."""
        client, mock_api = mock_client
        cal_df = pd.DataFrame({
            "cal_date": ["20240101", "20240102", "20240103", "20240104"],
            "is_open": [1, 1, 1, 0],
        })
        mock_api.trade_cal.return_value = cal_df

        result = client.get_trade_calendar("20240101", "20240104")
        assert result == ["20240101", "20240102", "20240103"]

        # Second call uses cache
        result2 = client.get_trade_calendar("20240101", "20240103")
        assert len(result2) == 3
        assert mock_api.trade_cal.call_count == 1  # Only called once
