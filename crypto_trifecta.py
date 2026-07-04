#!/usr/bin/env python3
"""
Crypto Trifecta Strategy
========================
Trades a $110 USDT wallet on Kraken, rotating between XRP, DOT, AVAX, ETH, and BTC.
$22.50 is the active trading capital (half of the $45 above gas reserve).
$65 gas reserve + $22.50 unused buffer held in USDT (Tether) at all times.
Rebalances every 12 hours. Only takes long positions on positive momentum (cash account,
no shorting). Uses trend confirmation (price above SMA) for higher upward-gain accuracy.
Only rebalances when allocation changes are significant to reduce fees.
"""

import os
import time
import logging
from datetime import datetime, timezone

import ccxt
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ── Logging ─────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Strategy constants ───────────────────────────────────────────────────────────
TOTAL_WALLET   = 110.00  # USDT – total wallet size
GAS_RESERVE    = 65.00   # USDT – always kept as USDT (gas / safety reserve)
BUFFER_USDT    = 22.50   # USDT – passive buffer, always kept in USDT
ACTIVE_CAPITAL = 22.50   # USDT – capital actively rotated across assets

# Sanity-check: the three buckets must add up to the total wallet
assert abs(GAS_RESERVE + BUFFER_USDT + ACTIVE_CAPITAL - TOTAL_WALLET) < 1e-6, (
    "Wallet constants do not sum to TOTAL_WALLET"
)

ASSETS  = ["XRP", "DOT", "AVAX", "ETH", "BTC"]
QUOTE   = "USDT"
PAIRS   = [f"{a}/{QUOTE}" for a in ASSETS]

REBALANCE_INTERVAL_H = 12          # hours between rebalance checks
REBALANCE_INTERVAL_S = REBALANCE_INTERVAL_H * 3600

SMA_PERIOD  = 20   # number of bars for the Simple Moving Average
OHLCV_TF    = "1h" # OHLCV timeframe
OHLCV_LIMIT = SMA_PERIOD + 1  # one extra bar as a safety margin

# Minimum fraction of ACTIVE_CAPITAL that an asset allocation must change before
# a rebalance is actually executed (reduces unnecessary fee-generating trades)
MIN_ALLOC_DELTA = 0.05  # 5 % → ~$1.13 per asset


# ── Exchange setup ───────────────────────────────────────────────────────────────

def build_exchange() -> ccxt.kraken:
    """Return an authenticated Kraken exchange instance using environment variables."""
    return ccxt.kraken({
        "apiKey": os.environ.get("KRAKEN_API_KEY", ""),
        "secret": os.environ.get("KRAKEN_API_SECRET", ""),
        "enableRateLimit": True,
    })


# ── Data helpers ─────────────────────────────────────────────────────────────────

def fetch_ohlcv(exchange: ccxt.Exchange, pair: str, limit: int = OHLCV_LIMIT) -> pd.DataFrame:
    """Fetch OHLCV candles for *pair* and return a tidy DataFrame."""
    raw = exchange.fetch_ohlcv(pair, timeframe=OHLCV_TF, limit=limit)
    df  = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.set_index("ts")


def sma(series: pd.Series, period: int = SMA_PERIOD) -> float:
    """Return the Simple Moving Average of the last *period* values in *series*."""
    return float(series.iloc[-period:].mean())


# ── Momentum / allocation ────────────────────────────────────────────────────────

def compute_momentum(exchange: ccxt.Exchange) -> dict[str, float]:
    """
    Evaluate positive momentum for every asset.

    An asset qualifies when its latest close price is **above** the 20-bar SMA
    (trend confirmation). The momentum score is the percentage gain above the SMA;
    assets below the SMA receive a score of 0.0 (no long position taken).
    """
    scores: dict[str, float] = {}
    for pair in PAIRS:
        asset = pair.split("/")[0]
        try:
            df    = fetch_ohlcv(exchange, pair)
            close = df["close"]
            price = float(close.iloc[-1])
            avg   = sma(close)
            if price > avg:
                scores[asset] = (price - avg) / avg
            else:
                scores[asset] = 0.0
            logger.info(
                "%s  price=%.6f  SMA(%d)=%.6f  score=%.6f",
                pair, price, SMA_PERIOD, avg, scores[asset],
            )
        except ccxt.BaseError as exc:
            logger.warning("Exchange error fetching %s: %s", pair, exc)
            scores[asset] = 0.0
        except Exception as exc:
            logger.warning("Unexpected error fetching %s: %s", pair, exc)
            scores[asset] = 0.0
    return scores


def compute_target_allocations(scores: dict[str, float]) -> dict[str, float]:
    """
    Convert momentum scores into target USDT allocations.

    Qualifying assets (score > 0) share ACTIVE_CAPITAL proportionally.
    If no asset qualifies, all ACTIVE_CAPITAL is kept in USDT (no long position).
    """
    total_score = sum(scores.values())
    if total_score == 0.0:
        logger.info("No positive-momentum assets – holding active capital in USDT.")
        return {asset: 0.0 for asset in ASSETS}
    return {
        asset: round(ACTIVE_CAPITAL * (score / total_score), 6)
        for asset, score in scores.items()
    }


# ── Portfolio state ──────────────────────────────────────────────────────────────

def fetch_current_allocations(exchange: ccxt.Exchange) -> dict[str, float]:
    """
    Return the current USDT-equivalent value held in each ASSET.

    Only inspects ACTIVE_CAPITAL assets; the GAS_RESERVE and BUFFER_USDT are
    untouched USDT and not considered here.
    """
    balance = exchange.fetch_balance()
    allocations: dict[str, float] = {}
    for asset in ASSETS:
        qty = float(balance.get(asset, {}).get("total", 0.0))
        if qty > 0.0:
            price = float(exchange.fetch_ticker(f"{asset}/{QUOTE}")["last"])
            allocations[asset] = qty * price
        else:
            allocations[asset] = 0.0
    return allocations


def allocation_delta_exceeds_threshold(
    current: dict[str, float],
    target:  dict[str, float],
) -> bool:
    """
    Return True when at least one asset's allocation would change by more than
    MIN_ALLOC_DELTA * ACTIVE_CAPITAL, warranting an actual rebalance.
    """
    threshold = MIN_ALLOC_DELTA * ACTIVE_CAPITAL
    for asset in ASSETS:
        delta = abs(target.get(asset, 0.0) - current.get(asset, 0.0))
        if delta >= threshold:
            logger.debug(
                "%s allocation delta $%.4f >= threshold $%.4f – rebalance needed",
                asset, delta, threshold,
            )
            return True
    return False


# ── Trade execution ──────────────────────────────────────────────────────────────

def rebalance(
    exchange:      ccxt.Exchange,
    current_alloc: dict[str, float],
    target_alloc:  dict[str, float],
) -> None:
    """
    Execute market orders to move from *current_alloc* → *target_alloc*.

    Sells are processed first to free USDT before buys are placed.
    Only long positions are taken; no short selling.
    """
    sells: dict[str, float] = {}
    buys:  dict[str, float] = {}

    for asset in ASSETS:
        diff = target_alloc.get(asset, 0.0) - current_alloc.get(asset, 0.0)
        if diff < -1e-9:
            sells[asset] = abs(diff)
        elif diff > 1e-9:
            buys[asset] = diff

    for asset, usdt_amount in sells.items():
        pair  = f"{asset}/{QUOTE}"
        price = float(exchange.fetch_ticker(pair)["last"])
        qty   = usdt_amount / price
        logger.info("SELL  %.8f %s  (~$%.4f USDT)", qty, asset, usdt_amount)
        try:
            exchange.create_market_sell_order(pair, qty)
        except ccxt.BaseError as exc:
            logger.error("Sell order failed for %s: %s", asset, exc)

    for asset, usdt_amount in buys.items():
        pair  = f"{asset}/{QUOTE}"
        price = float(exchange.fetch_ticker(pair)["last"])
        qty   = usdt_amount / price
        logger.info("BUY   %.8f %s  (~$%.4f USDT)", qty, asset, usdt_amount)
        try:
            exchange.create_market_buy_order(pair, qty)
        except ccxt.BaseError as exc:
            logger.error("Buy order failed for %s: %s", asset, exc)


# ── Main loop ────────────────────────────────────────────────────────────────────

def run() -> None:
    """
    Strategy main loop.

    Evaluates momentum every REBALANCE_INTERVAL_H hours and rebalances only
    when at least one asset allocation would change by more than the minimum
    threshold, keeping transaction costs low.
    """
    exchange = build_exchange()
    logger.info(
        "Crypto Trifecta started | total=$%.2f | active=$%.2f | "
        "gas_reserve=$%.2f | buffer=$%.2f | interval=%dh",
        TOTAL_WALLET, ACTIVE_CAPITAL, GAS_RESERVE, BUFFER_USDT,
        REBALANCE_INTERVAL_H,
    )

    while True:
        cycle_start = datetime.now(timezone.utc)
        logger.info(
            "── Rebalance cycle %s ──────────────────────────────────",
            cycle_start.strftime("%Y-%m-%d %H:%M:%S UTC"),
        )

        try:
            scores        = compute_momentum(exchange)
            target_alloc  = compute_target_allocations(scores)
            current_alloc = fetch_current_allocations(exchange)

            logger.info("Current allocations (USDT): %s", current_alloc)
            logger.info("Target  allocations (USDT): %s", target_alloc)

            if allocation_delta_exceeds_threshold(current_alloc, target_alloc):
                logger.info("Allocation delta exceeds threshold – rebalancing.")
                rebalance(exchange, current_alloc, target_alloc)
            else:
                logger.info("Allocation delta below threshold – no rebalance needed.")

        except Exception as exc:
            logger.exception("Unexpected error during rebalance cycle: %s", exc)

        elapsed_s = (datetime.now(timezone.utc) - cycle_start).total_seconds()
        sleep_s   = max(0.0, REBALANCE_INTERVAL_S - elapsed_s)
        logger.info(
            "Cycle complete. Next rebalance in %.0f s (%.1f h).",
            sleep_s, sleep_s / 3600,
        )
        time.sleep(sleep_s)


if __name__ == "__main__":
    run()
