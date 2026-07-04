"""
Unit tests for crypto_trifecta.py

All tests exercise pure-logic functions only; no live exchange calls are made.
"""

import pytest
import ccxt
import pandas as pd
from unittest.mock import MagicMock, patch, call

import crypto_trifecta as ct


# ── Wallet constants ─────────────────────────────────────────────────────────────

class TestWalletConstants:
    def test_buckets_sum_to_total(self):
        """GAS_RESERVE + BUFFER_USDT + ACTIVE_CAPITAL must equal TOTAL_WALLET."""
        assert abs(ct.GAS_RESERVE + ct.BUFFER_USDT + ct.ACTIVE_CAPITAL - ct.TOTAL_WALLET) < 1e-6

    def test_active_capital_is_half_of_surplus(self):
        """Active capital should be half the $45 above the $65 gas reserve."""
        surplus = ct.TOTAL_WALLET - ct.GAS_RESERVE  # 110 - 65 = 45
        assert abs(ct.ACTIVE_CAPITAL - surplus / 2) < 1e-6

    def test_total_wallet_is_110(self):
        assert ct.TOTAL_WALLET == 110.00

    def test_gas_reserve_is_65(self):
        assert ct.GAS_RESERVE == 65.00

    def test_buffer_and_active_are_22_50(self):
        assert ct.BUFFER_USDT == 22.50
        assert ct.ACTIVE_CAPITAL == 22.50

    def test_assets_list(self):
        assert set(ct.ASSETS) == {"XRP", "DOT", "AVAX", "ETH", "BTC"}
        assert len(ct.ASSETS) == 5

    def test_rebalance_interval(self):
        assert ct.REBALANCE_INTERVAL_H == 12
        assert ct.REBALANCE_INTERVAL_S == 12 * 3600


# ── sma() ────────────────────────────────────────────────────────────────────────

class TestSMA:
    def test_basic_average(self):
        series = pd.Series([float(i) for i in range(1, 21)])  # 1..20, mean=10.5
        assert abs(ct.sma(series, period=20) - 10.5) < 1e-9

    def test_uses_only_last_n_bars(self):
        """SMA must use the most recent *period* bars, ignoring older ones."""
        series = pd.Series([1.0] * 15 + [200.0] * 5)
        assert abs(ct.sma(series, period=5) - 200.0) < 1e-9

    def test_constant_series(self):
        series = pd.Series([42.0] * 20)
        assert abs(ct.sma(series, period=20) - 42.0) < 1e-9

    def test_single_bar(self):
        series = pd.Series([99.0])
        assert abs(ct.sma(series, period=1) - 99.0) < 1e-9

    def test_default_period_is_sma_period_constant(self):
        series = pd.Series([float(i) for i in range(1, ct.SMA_PERIOD + 1)])
        expected = sum(range(1, ct.SMA_PERIOD + 1)) / ct.SMA_PERIOD
        assert abs(ct.sma(series) - expected) < 1e-9


# ── compute_target_allocations() ─────────────────────────────────────────────────

class TestComputeTargetAllocations:
    def test_all_zero_scores_returns_zero_allocations(self):
        scores = {a: 0.0 for a in ct.ASSETS}
        allocs = ct.compute_target_allocations(scores)
        assert all(v == 0.0 for v in allocs.values())
        assert set(allocs.keys()) == set(ct.ASSETS)

    def test_single_qualifying_asset_gets_full_active_capital(self):
        scores = {a: 0.0 for a in ct.ASSETS}
        scores["ETH"] = 0.05
        allocs = ct.compute_target_allocations(scores)
        assert abs(allocs["ETH"] - ct.ACTIVE_CAPITAL) < 1e-3
        for a in ct.ASSETS:
            if a != "ETH":
                assert allocs[a] == 0.0

    def test_two_equal_scores_split_capital_evenly(self):
        scores = {a: 0.0 for a in ct.ASSETS}
        scores["BTC"] = 0.10
        scores["ETH"] = 0.10
        allocs = ct.compute_target_allocations(scores)
        assert abs(allocs["BTC"] - ct.ACTIVE_CAPITAL / 2) < 1e-3
        assert abs(allocs["ETH"] - ct.ACTIVE_CAPITAL / 2) < 1e-3

    def test_proportional_allocation(self):
        scores = {"XRP": 0.0, "DOT": 0.0, "AVAX": 0.0, "ETH": 1.0, "BTC": 3.0}
        allocs = ct.compute_target_allocations(scores)
        assert abs(allocs["ETH"] - ct.ACTIVE_CAPITAL * 0.25) < 1e-3
        assert abs(allocs["BTC"] - ct.ACTIVE_CAPITAL * 0.75) < 1e-3
        assert allocs["XRP"] == 0.0
        assert allocs["DOT"] == 0.0
        assert allocs["AVAX"] == 0.0

    def test_total_allocation_equals_active_capital(self):
        scores = {"XRP": 0.01, "DOT": 0.02, "AVAX": 0.03, "ETH": 0.04, "BTC": 0.05}
        allocs = ct.compute_target_allocations(scores)
        assert abs(sum(allocs.values()) - ct.ACTIVE_CAPITAL) < 1e-3

    def test_all_assets_qualifying(self):
        scores = {a: 0.05 for a in ct.ASSETS}
        allocs = ct.compute_target_allocations(scores)
        per_asset = ct.ACTIVE_CAPITAL / len(ct.ASSETS)
        for a in ct.ASSETS:
            assert abs(allocs[a] - per_asset) < 1e-3


# ── allocation_delta_exceeds_threshold() ─────────────────────────────────────────

class TestAllocationDeltaExceedsThreshold:
    def _zero(self):
        return {a: 0.0 for a in ct.ASSETS}

    def test_no_change_returns_false(self):
        current = {a: 5.0 for a in ct.ASSETS}
        assert not ct.allocation_delta_exceeds_threshold(current, current.copy())

    def test_large_change_returns_true(self):
        current = self._zero()
        target  = {a: ct.ACTIVE_CAPITAL for a in ct.ASSETS}
        assert ct.allocation_delta_exceeds_threshold(current, target)

    def test_change_below_threshold_returns_false(self):
        threshold = ct.MIN_ALLOC_DELTA * ct.ACTIVE_CAPITAL
        current = {a: 5.0 for a in ct.ASSETS}
        target  = {a: 5.0 for a in ct.ASSETS}
        target["XRP"] = 5.0 + threshold * 0.5  # half the threshold
        assert not ct.allocation_delta_exceeds_threshold(current, target)

    def test_change_exactly_at_threshold_returns_true(self):
        threshold = ct.MIN_ALLOC_DELTA * ct.ACTIVE_CAPITAL
        current = {a: 5.0 for a in ct.ASSETS}
        target  = {a: 5.0 for a in ct.ASSETS}
        target["XRP"] = 5.0 + threshold  # exactly the threshold
        assert ct.allocation_delta_exceeds_threshold(current, target)

    def test_missing_asset_in_target_treated_as_zero(self):
        current = {"XRP": 5.0, "DOT": 0.0, "AVAX": 0.0, "ETH": 0.0, "BTC": 0.0}
        target  = {"DOT": 0.0, "AVAX": 0.0, "ETH": 0.0, "BTC": 0.0}
        # XRP current=5.0, target missing (treated as 0.0) → delta=5.0 > threshold
        assert ct.allocation_delta_exceeds_threshold(current, target)


# ── compute_momentum() – mocked exchange ─────────────────────────────────────────

class TestComputeMomentum:
    def _make_ohlcv(self, prices: list[float]) -> list[list]:
        """Build a minimal raw OHLCV list from a list of close prices."""
        return [[i * 3_600_000, p, p, p, p, 1000.0] for i, p in enumerate(prices)]

    def test_price_above_sma_returns_positive_score(self):
        exchange = MagicMock()
        # 20 bars of 1.0, then final bar at 2.0 → price > SMA
        prices = [1.0] * 20 + [2.0]
        exchange.fetch_ohlcv.return_value = self._make_ohlcv(prices)
        scores = ct.compute_momentum(exchange)
        for asset in ct.ASSETS:
            assert scores[asset] > 0.0, f"Expected positive score for {asset}"

    def test_price_below_sma_returns_zero_score(self):
        exchange = MagicMock()
        # 20 bars of 2.0, then drop to 1.0 → price < SMA
        prices = [2.0] * 20 + [1.0]
        exchange.fetch_ohlcv.return_value = self._make_ohlcv(prices)
        scores = ct.compute_momentum(exchange)
        for asset in ct.ASSETS:
            assert scores[asset] == 0.0, f"Expected zero score for {asset}"

    def test_price_equal_to_sma_returns_zero_score(self):
        exchange = MagicMock()
        prices = [1.0] * 21  # price == SMA → not strictly above
        exchange.fetch_ohlcv.return_value = self._make_ohlcv(prices)
        scores = ct.compute_momentum(exchange)
        for asset in ct.ASSETS:
            assert scores[asset] == 0.0

    def test_exchange_error_returns_zero_score(self):
        exchange = MagicMock()
        exchange.fetch_ohlcv.side_effect = ccxt.NetworkError("timeout")
        scores = ct.compute_momentum(exchange)
        for asset in ct.ASSETS:
            assert scores[asset] == 0.0

    def test_returns_score_for_all_assets(self):
        exchange = MagicMock()
        prices = [1.0] * 21
        exchange.fetch_ohlcv.return_value = self._make_ohlcv(prices)
        scores = ct.compute_momentum(exchange)
        assert set(scores.keys()) == set(ct.ASSETS)


# ── rebalance() – mocked exchange ────────────────────────────────────────────────

class TestRebalance:
    def test_sells_then_buys(self):
        exchange = MagicMock()
        exchange.fetch_ticker.return_value = {"last": 100.0}

        current = {"XRP": 10.0, "DOT": 0.0, "AVAX": 0.0, "ETH": 0.0, "BTC": 0.0}
        target  = {"XRP": 0.0,  "DOT": 5.0, "AVAX": 0.0, "ETH": 0.0, "BTC": 0.0}

        ct.rebalance(exchange, current, target)

        exchange.create_market_sell_order.assert_called_once()
        exchange.create_market_buy_order.assert_called_once()

        sell_args = exchange.create_market_sell_order.call_args[0]
        buy_args  = exchange.create_market_buy_order.call_args[0]
        assert sell_args[0] == "XRP/USDT"
        assert buy_args[0]  == "DOT/USDT"

    def test_no_trades_when_no_diff(self):
        exchange = MagicMock()
        allocs = {a: 0.0 for a in ct.ASSETS}
        ct.rebalance(exchange, allocs, allocs.copy())
        exchange.create_market_sell_order.assert_not_called()
        exchange.create_market_buy_order.assert_not_called()

    def test_buy_order_failure_is_handled(self):
        exchange = MagicMock()
        exchange.fetch_ticker.return_value = {"last": 50.0}
        exchange.create_market_buy_order.side_effect = ccxt.InsufficientFunds("low")

        current = {a: 0.0 for a in ct.ASSETS}
        target  = {a: 0.0 for a in ct.ASSETS}
        target["BTC"] = 5.0

        # Should not raise; error is caught and logged
        ct.rebalance(exchange, current, target)

    def test_sell_order_failure_is_handled(self):
        exchange = MagicMock()
        exchange.fetch_ticker.return_value = {"last": 50.0}
        exchange.create_market_sell_order.side_effect = ccxt.ExchangeError("err")

        current = {a: 0.0 for a in ct.ASSETS}
        current["ETH"] = 5.0
        target  = {a: 0.0 for a in ct.ASSETS}

        ct.rebalance(exchange, current, target)  # must not raise


# ── fetch_current_allocations() – mocked exchange ────────────────────────────────

class TestFetchCurrentAllocations:
    def test_zero_qty_returns_zero_usdt(self):
        exchange = MagicMock()
        exchange.fetch_balance.return_value = {a: {"total": 0.0} for a in ct.ASSETS}
        allocs = ct.fetch_current_allocations(exchange)
        assert all(v == 0.0 for v in allocs.values())
        exchange.fetch_ticker.assert_not_called()

    def test_nonzero_qty_multiplied_by_price(self):
        exchange = MagicMock()
        balance = {a: {"total": 0.0} for a in ct.ASSETS}
        balance["ETH"] = {"total": 2.0}
        exchange.fetch_balance.return_value = balance
        exchange.fetch_ticker.return_value = {"last": 3000.0}
        allocs = ct.fetch_current_allocations(exchange)
        assert abs(allocs["ETH"] - 6000.0) < 1e-6

    def test_missing_asset_in_balance_treated_as_zero(self):
        exchange = MagicMock()
        exchange.fetch_balance.return_value = {}
        allocs = ct.fetch_current_allocations(exchange)
        assert all(v == 0.0 for v in allocs.values())
