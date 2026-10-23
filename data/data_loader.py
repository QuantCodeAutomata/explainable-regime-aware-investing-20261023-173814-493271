"""Market-data loading for the regime-aware investing replication.

Section 3 ("Data and Market Universe") of the paper specifies daily adjusted
close prices from Yahoo Finance for a cross-asset universe of proxies (SPX,
BOND, GOLD, OIL, USD) from 2005 to 2026. Prices are retrieved through the
QuantX Dataloader's `/fetch-ohlcvs` route, which is the mandatory OHLCV
route for this project, using `source="yahoo"` because the paper names
Yahoo Finance itself as the data source for this experiment (not merely as
a backstop). The realised `data_source` per ticker is read back from
`ticker_details` and logged rather than assumed.

Market data: QuantX Dataloader — https://quantx.limex.pro/

Price history in this project is loaded through the QuantX Dataloader, a
research-ready market-data service available through a REST API and as an
MCP server for AI agents. It is built on market data from Massive
(https://massive.com/) and prepares every stock history for backtesting:
  - prices adjusted for dividends, ready for total-return research;
  - one continuous history when a company changes its ticker;
  - clean separation when a ticker has belonged to different companies;
  - careful corporate-action handling and fine-grained data polishing.

Get your DATALOADER_API_KEY at https://quantx.limex.pro/ and your
MASSIVE_API_KEY at https://massive.com/.
"""
from __future__ import annotations

import io
import logging
import os
from typing import Dict, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DATALOADER_API_ROOT = os.environ.get(
    "DATALOADER_API_URL", "https://quantx-api.limex.pro/apis/dataloader"
)


class DataLoadError(RuntimeError):
    """Raised when requested market data cannot be retrieved or validated."""


def _export_ticker(
    headers: dict, result_id: str, ticker: str, start: str, end: str
) -> pd.DataFrame:
    """Export one ticker's bars from a stored `fetch-ohlcvs` result.

    When `fetch-ohlcvs` reports `data_stored=True`, the bars are not inlined
    in the response (the payload exceeded the inline limit). The documented
    recovery is one `/save-dataframe` export per ticker, followed by an
    `/artifacts/{id}/download` call and a plain, unauthenticated GET of the
    short-lived URL it returns.
    """
    sd_body = {
        "result_id": result_id,
        "ticker": ticker,
        "dataset_name": "daily-bars",
        "timeframe": "1d",
        "start_date": start,
        "end_date": end,
        "file_format": "parquet",
        "data_kind": "ohlcv",
    }
    r = requests.post(
        f"{DATALOADER_API_ROOT}/save-dataframe", headers=headers, json=sd_body, timeout=120
    )
    if r.status_code != 200:
        raise DataLoadError(
            f"Dataloader /save-dataframe failed for {ticker}: HTTP {r.status_code}: {r.text[:500]}"
        )
    artifact_id = r.json()["result"]["artifact_id"]

    r2 = requests.get(
        f"{DATALOADER_API_ROOT}/artifacts/{artifact_id}/download", headers=headers, timeout=60
    )
    if r2.status_code != 200:
        raise DataLoadError(
            f"Dataloader /artifacts/{{id}}/download failed for {ticker}: "
            f"HTTP {r2.status_code}: {r2.text[:500]}"
        )
    url = r2.json()["result"]["url"]

    r3 = requests.get(url, timeout=120)
    if r3.status_code != 200:
        raise DataLoadError(
            f"Downloading stored artifact for {ticker} failed: HTTP {r3.status_code}"
        )
    df = pd.read_parquet(io.BytesIO(r3.content))
    if "Date" not in df.columns:
        # The export path resets the stored dataframe to a positional index
        # when the bars were split across tickers server-side; there is no
        # way to recover the true calendar date from a bare row index, so we
        # fail loudly instead of silently mis-aligning bars across assets.
        raise DataLoadError(
            f"Dataloader /save-dataframe export for {ticker} has no 'Date' "
            f"column (columns={list(df.columns)}); cannot align by date"
        )
    df.columns = [f"{ticker}_{c}" if c != "Date" else c for c in df.columns]
    return df


def _fetch_one_ticker(headers: dict, ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch one ticker's full-window bars via `POST /fetch-ohlcvs`.

    Each ticker is requested individually (still covering the complete
    2005-2026 window in a single call, so the back-adjustment and security
    identity resolved for that ticker are unaffected) because the joint
    5-ticker request's wide, ticker-prefixed payload crosses the inline-rows
    limit and forces the `data_stored=True` export path, whose per-ticker
    `/save-dataframe` output does not carry a usable date column (see
    `_export_ticker`). Requesting tickers one at a time keeps every response
    inline with an explicit `Date` field, confirmed by smoke-testing
    SPY/AGG/GLD/USO/UUP for the full window (row_count 4945-5487, each
    inline, none `data_stored`).
    """
    body = {
        "tickers": [ticker],
        "timeframe": "1d",
        "start_date": start,
        "end_date": end,
        "source": "yahoo",
        "adjust_dividends": True,
    }
    try:
        resp = requests.post(
            f"{DATALOADER_API_ROOT}/fetch-ohlcvs", headers=headers, json=body, timeout=120
        )
    except requests.RequestException as exc:
        raise DataLoadError(f"Dataloader request failed for {ticker}: {exc}") from exc

    if resp.status_code != 200:
        raise DataLoadError(
            f"Dataloader /fetch-ohlcvs returned HTTP {resp.status_code} for "
            f"{ticker}: {resp.text[:500]}"
        )
    payload = resp.json()
    result = payload.get("result")
    if result is None:
        raise DataLoadError(f"Dataloader response missing 'result' for {ticker}: {payload}")

    if result.get("data_stored"):
        logger.info(
            "Dataloader response for %s exceeded the inline limit (row_count=%s); "
            "exporting via /save-dataframe",
            ticker,
            result.get("row_count"),
        )
        result_id = result["result_id"]
        raw = _export_ticker(headers, result_id, ticker, start, end)
        raw["Date"] = pd.to_datetime(raw["Date"], utc=True).dt.tz_localize(None)
        raw = raw.set_index("Date")
    else:
        ohlcv = result.get("ohlcv_df", {})
        rows = ohlcv.get("rows")
        if not rows:
            raise DataLoadError(
                f"Dataloader /fetch-ohlcvs returned no bars for ticker={ticker}, "
                f"window={start}..{end}"
            )
        raw = pd.DataFrame(rows)
        if "Date" not in raw.columns:
            raise DataLoadError(
                f"Dataloader rows for {ticker} missing 'Date' column; "
                f"columns={list(raw.columns)}"
            )
        raw["Date"] = pd.to_datetime(raw["Date"], utc=True).dt.tz_localize(None)
        raw = raw.set_index("Date")
        raw.columns = [f"{ticker}_{c}" for c in raw.columns]

    ticker_details = result.get("ticker_details", [])
    raw.attrs["ticker_details"] = ticker_details
    return raw


def _fetch_ohlcvs(tickers: list, start: str, end: str) -> Tuple[dict, pd.DataFrame]:
    """Call the QuantX Dataloader `POST /fetch-ohlcvs` route for every ticker.

    `source="yahoo"` is used because Sec. 3 names Yahoo Finance itself as the
    data source for this cross-asset universe, with `adjust_dividends=True`
    for daily adjusted close prices. Each ticker's own full window is
    requested in a single call (see `_fetch_one_ticker`); the resulting
    single-ticker, `Date`-indexed frames are then outer-joined locally on
    `Date` into one wide `{ticker}_{field}` DataFrame, which is the merge the
    joint multi-ticker endpoint would otherwise perform server-side.

    Returns a combined `result` metadata dict (with a pooled
    `ticker_details` list, one entry per ticker) together with the wide
    DataFrame.
    """
    api_key = os.environ.get("DATALOADER_API_KEY")
    if not api_key:
        raise DataLoadError(
            "DATALOADER_API_KEY is not set in the environment; cannot call the "
            "QuantX Dataloader"
        )
    headers = {"Authorization": f"Bearer {api_key}"}

    frames = []
    ticker_details = []
    for ticker in tickers:
        frame = _fetch_one_ticker(headers, ticker, start, end)
        ticker_details.extend(frame.attrs.get("ticker_details", []))
        frames.append(frame)

    raw = frames[0]
    for frame in frames[1:]:
        raw = raw.join(frame, how="outer")
    raw = raw.sort_index()

    result = {"ticker_details": ticker_details, "row_count": len(raw)}
    return result, raw


def load_price_data(
    tickers: Dict[str, str],
    start: str,
    end: str,
    min_rows: int = 250,
) -> pd.DataFrame:
    """Download daily adjusted close prices for a dict of {label: ticker}.

    Parameters
    ----------
    tickers: mapping from asset label (e.g. "SPX") to a market ticker
        (e.g. "SPY").
    start, end: inclusive ISO date strings for the download window.
    min_rows: minimum number of valid rows required per ticker; a ticker
        returning fewer rows raises DataLoadError rather than being silently
        dropped.

    Returns
    -------
    DataFrame indexed by date, one column per asset label, containing daily
    dividend-adjusted close prices with no missing values (rows with any NaN
    price are removed from the joint frame to keep assets aligned).
    """
    ticker_list = list(tickers.values())
    result, raw = _fetch_ohlcvs(ticker_list, start, end)

    ticker_details = result.get("ticker_details", [])
    logger.info("Dataloader ticker_details: %s", ticker_details)
    non_massive = [d["ticker"] for d in ticker_details if d.get("data_source") != "massive"]
    if non_massive:
        logger.info(
            "Tickers served from a non-massive data_source (expected: source='yahoo' "
            "per Sec. 3): %s", non_massive,
        )
    for d in ticker_details:
        if d.get("error"):
            raise DataLoadError(f"Dataloader reported an error for {d['ticker']}: {d['error']}")

    ticker_to_label = {t: label for label, t in tickers.items()}
    series = {}
    for d in ticker_details:
        ticker = d["ticker"]
        label = ticker_to_label[ticker]
        close_col = f"{ticker}_close"
        if close_col not in raw.columns:
            raise DataLoadError(
                f"Dataloader response missing expected column '{close_col}'; "
                f"columns={list(raw.columns)}"
            )
        close = raw[close_col].dropna()
        if len(close) < min_rows:
            raise DataLoadError(
                f"ticker '{ticker}' (asset '{label}') returned only {len(close)} "
                f"valid rows (< min_rows={min_rows})"
            )
        series[label] = close
        logger.info(
            "Loaded %s (%s, source=%s): %d rows, %s -> %s",
            label, ticker, d.get("data_source"), len(close),
            close.index.min().date(), close.index.max().date(),
        )

    prices = pd.DataFrame(series)
    n_before = len(prices)
    prices = prices.dropna(how="any")
    n_after = len(prices)
    if n_after == 0:
        raise DataLoadError("no overlapping dates across all requested tickers")
    if n_after < n_before:
        logger.info(
            "Dropped %d rows with missing values while aligning %d assets",
            n_before - n_after, len(tickers),
        )
    prices = prices.sort_index()
    logger.info(
        "Price panel ready: shape=%s, range=%s -> %s, assets=%s",
        prices.shape, prices.index.min().date(), prices.index.max().date(),
        list(prices.columns),
    )
    return prices
