### favorite bot

# region imports
from AlgorithmImports import *
# endregion


class CryptoTrifectaAlgorithm(QCAlgorithm):
    """
    Crypto Trifecta Strategy:
    Trades a $110 USDT wallet on Kraken, rotating between XRP, DOT, AVAX, ETH, and BTC.
    $22.50 is the active trading capital (half of the $45 above gas reserve).
    $65 gas reserve + $22.50 unused buffer held in USDT (Tether) at all times.
    Rebalances every 12 hours. Only takes long positions on positive momentum (cash account, no shorting).
    Uses trend confirmation (price above SMA) for higher upward-gain accuracy.
    Only rebalances when allocation changes are significant to reduce fees.
    """

    def initialize(self) -> None:
        self.set_start_date(2020, 1, 1)
        self.set_end_date(2026, 1, 1)
        self.set_account_currency("USDT", 1000)
        self.set_brokerage_model(BrokerageName.KRAKEN, AccountType.CASH)

        # Set time zone to UTC (crypto markets trade 24/7)
        self.set_time_zone(TimeZones.UTC)

        # Allow fractional crypto trading for a small wallet
        self.settings.minimum_order_margin_portfolio_percentage = 0
        # Larger buffer to prevent insufficient buying power errors on small accounts
        self.settings.free_portfolio_value_percentage = 0.08

        # The assets in the rotation (BTC included)
        self._symbols = {}
        self._securities = {}
        for ticker in ["XRP", "DOT", "AVAX", "ETH", "BTC"]:
            crypto = self.add_crypto(f"{ticker}USDT", Resolution.HOUR, Market.KRAKEN)
            self._symbols[ticker] = crypto.symbol
            self._securities[ticker] = crypto

        # Optimizable parameters (with defaults)
        self._lookback_hours = int(self.get_parameter("lookback_hours", 72))
        self._rebalance_hours = int(self.get_parameter("rebalance_hours", 12))
        self._momentum_threshold = float(self.get_parameter("momentum_threshold", 0.01))

        # Gas/fee reserve kept in USDT (Tether) at all times
        self._gas_reserve = 65

        # Allocation factor — halved from full trading capital for slower, safer exposure
        self._allocation_factor = 0.50

        # Minimum allocation weight — skip assets whose position would be too small
        self._min_weight = 0.10

        # Rebalance tolerance — skip rebalance if target weights haven't changed enough
        self._rebalance_tolerance = 0.05

        # Track current target weights to avoid unnecessary rebalancing
        self._current_weights = {}

        # Schedule rebalance every 12 hours starting at midnight
        self.schedule.on(
            self.date_rules.every_day(),
            self.time_rules.every(timedelta(hours=self._rebalance_hours)),
            self._rebalance,
        )

    def _rebalance(self) -> None:
        """
        Evaluate momentum (rate of change over lookback) for each asset with trend confirmation.
        Only assets with positive momentum above threshold AND price above SMA qualify.
        Allocate half of trading capital to the best performer.
        Cash account means no shorting — if nothing qualifies, stay in USDT (Tether).
        """
        # Gather recent history for all symbols in one batch request
        symbols = list(self._symbols.values())
        history = self.history(symbols, self._lookback_hours, Resolution.HOUR)

        if history.empty:
            self.log("No history data available, skipping rebalance.")
            return

        # Calculate rate of change and trend confirmation for each symbol
        momentum_scores = {}
        for ticker, symbol in self._symbols.items():
            if symbol not in history.index.get_level_values(0):
                continue
            closes = history.loc[symbol]["close"]
            if len(closes) < 2:
                continue
            old_price = closes.iloc[0]
            new_price = closes.iloc[-1]
            sma = closes.mean()

            if old_price > 0:
                roc = (new_price - old_price) / old_price
            else:
                roc = 0.0

            # Trend confirmation: only qualify if positive momentum AND price above SMA
            # This filters out short-term bounces in longer-term downtrends
            if roc > self._momentum_threshold and new_price > sma:
                momentum_scores[ticker] = roc

        if not momentum_scores:
            self.log(f"No asset qualifies (momentum > {self._momentum_threshold} and price > SMA). Staying in USDT.")
            self.liquidate()
            self._current_weights = {}
            return

        # Select the single best asset (concentrate for accuracy)
        best_ticker = max(momentum_scores, key=lambda k: momentum_scores[k])
        best_score = momentum_scores[best_ticker]
        best_symbol = self._symbols[best_ticker]

        # Calculate dynamic allocation: half of trading capital above gas reserve
        portfolio_value = self.portfolio.total_portfolio_value
        trading_capital = portfolio_value - self._gas_reserve

        if trading_capital <= 0:
            self.log(f"Insufficient trading capital above gas reserve. Portfolio: {portfolio_value}, Reserve: {self._gas_reserve}")
            return

        # Base weight: half the trading capital, with fee buffer
        base_weight = (trading_capital / portfolio_value) * self._allocation_factor * 0.95

        if base_weight < self._min_weight:
            self.log(f"Allocation weight {base_weight:.4f} below minimum {self._min_weight}. Skipping.")
            return

        # Check if rebalance is needed
        if self._current_weights.get(best_ticker, 0) > 0 and \
           abs(base_weight - self._current_weights.get(best_ticker, 0)) < self._rebalance_tolerance:
            self.log(f"Holding {best_ticker} (momentum={best_score:.4f}). Within tolerance, no change.")
            return

        self.log(f"Rotating into {best_ticker} (momentum={best_score:.4f}, weight={base_weight:.4f}). All scores: {momentum_scores}")

        # Liquidate and buy the best performer
        self.set_holdings([PortfolioTarget(best_symbol, base_weight)], liquidate_existing_holdings=True)
        self._current_weights = {best_ticker: base_weight}
