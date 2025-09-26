import logging
import math
import warnings
from typing import Tuple, Literal

warnings.simplefilter(action="ignore", category=FutureWarning)

from trader.config import Config
from trader.core.base_robot import *

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class PiggyBackSniper:
    """
    A trading bot that scouts for new tokens, opens positions,
    and applies a “peak-and-retrace” partial selling strategy with trailing stops.

    The base currency is USDC. We keep track of actual token holdings
    in self.holdings, a dict keyed by the token address, including
    USDC itself (via USDC_MINT_ADDRESS).
    """

    def __init__(
        self,
        birdeye_api_key: str,
        private_key_base58: str,
        wallet_address: str,
        rpc_url: str = Config.RPC_URL,
        starting_usdc_balance: float = 1000.0,
        max_open_trades: int = 10,
        usd_per_trade: float = 5.0,
        scout_interval: Tuple[int, Literal["seconds", "minutes", "hours"]] = (
            5,
            "minutes",
        ),
        position_check_interval: int = 30,
        analyze_potential_entries_interval: int = 60,
        paper_trading: bool = False,
        table_name: str = "trades",
        bot_type: str = "sniper",
        mc_bounds: Tuple[float, float] = (50_000, 20_000_000),
        min_trades_in_30m: int = 30,
        max_drawdown_allowed: float = 70.0,
        top10_holder_percent_limit: float = 0.30,
        creation_max_days_ago: int = None,
        creation_min_days_ago: int = 90,  # token must be created within these many days
        creation_max_hours_ago: int = 2,  # token must NOT be created too recently, e.g. last 2 hours
        trade_opened_less_than_x_mins_ago: int = 60,  # trade must be opened within these many minutes
        random_sample_size: int = 10,
        retention_probability: float = 0.95,
        next_tradable_hours: int = 24,
        sma_window_hours: int = 24,
    ):
        """
        Initialize the PiggyBackSniper with necessary configuration.
        """
        self.bot = BaseBot(
            strategy_function=self.apply_peak_and_retrace_strategy,
            wallet_address=wallet_address,
            birdeye_api_key=birdeye_api_key,
            private_key_base58=private_key_base58,
            rpc_url=rpc_url,
            starting_usdc_balance=starting_usdc_balance,
            max_open_trades=max_open_trades,
            usd_per_trade=usd_per_trade,
            scout_interval=scout_interval,
            position_check_interval=position_check_interval,
            analyze_potential_entries_interval=analyze_potential_entries_interval,
            paper_trading=paper_trading,
            table_name=table_name,
            bot_type=bot_type,
            mc_bounds=mc_bounds,
            min_trades_in_30m=min_trades_in_30m,
            max_drawdown_allowed=max_drawdown_allowed,
            top10_holder_percent_limit=top10_holder_percent_limit,
            creation_max_days_ago=creation_max_days_ago,
            creation_min_days_ago=creation_min_days_ago,
            creation_max_hours_ago=creation_max_hours_ago,
            trade_opened_less_than_x_mins_ago=trade_opened_less_than_x_mins_ago,
            random_sample_size=random_sample_size,
            retention_probability=retention_probability,
            next_tradable_hours=next_tradable_hours,
            sma_window_hours=sma_window_hours,
        )

    def run(self):
        self.bot.run()

    # ---------------------------------------------------------
    # PEAK-AND-RETRACE STRATEGY
    # ---------------------------------------------------------

    def apply_peak_and_retrace_strategy(self, idx: int, current_price: float):
        """
        Applies multi-threshold partial-selling strategy and trailing stops
        for the position at DataFrame index idx.
        Removes the position from the DataFrame if fully closed.
        """
        with self.bot.positions_lock:
            if idx not in self.bot.positions_df.index:
                return  # already removed somewhere else
            row = self.bot.positions_df.loc[idx]

        if row["status"] != "open":
            return

        entry_price = row["entry_price"]
        original_amount = row["entry_amount"]
        partial_sold_cum = row["partial_sold_cumulative"]
        realized_profit_usd = row["realized_profit_usd"]
        max_price = row["max_price"]
        stoploss_count = row.get("stoploss_count", 0)

        # Update max_price if current is higher
        if current_price > max_price:
            max_price = current_price

        (current_price / entry_price) if entry_price > 0 else 0
        max_factor = (max_price / entry_price) if entry_price > 0 else 0

        # 1) Partial-sell thresholds
        partial_sells_table = [
            # (1.3, 0.05), # worth?
            (1.5, 0.10),
            (1.75, 0.15),
            (2.0, 0.20),
            (2.5, 0.25),
            (3.0, 0.30),
            (3.5, 0.35),
            (4.0, 0.40),
            (4.5, 0.45),
            (5.0, 0.50),
            (6.0, 0.55),
            (7.0, 0.60),
            (8.0, 0.65),
            (9.0, 0.70),
            (10.0, 0.75),
            (15.0, 0.80),
            (20.0, 0.90),
            (30.0, 1.0),
        ]

        token_addr = row["token_address"]

        # Determine total fraction to sell in one operation
        for threshold_factor, fraction_sold_target in reversed(partial_sells_table):
            if (
                max_factor >= threshold_factor
                and partial_sold_cum < fraction_sold_target
            ):
                # Sell the difference
                fraction_to_sell_now = fraction_sold_target - partial_sold_cum
                tokens_to_sell = original_amount * fraction_to_sell_now
                if tokens_to_sell > 0:
                    logging.info(
                        f"{GREEN}Crossed {threshold_factor}× => partial sells up to "
                        f"{fraction_sold_target * 100:.0f}% total.{RESET}"
                    )
                    part_profit = self.bot.sell_token(
                        token_address=token_addr,
                        token_amount=tokens_to_sell,
                        partial_sold_cumulative=float(fraction_sold_target),
                        realized_pnl=float(realized_profit_usd),
                        stoploss_price=None,
                        max_recorded_price=float(max_price),
                    )
                    if realized_profit_usd is None or math.isnan(realized_profit_usd):
                        realized_profit_usd = 0.0
                    realized_profit_usd += part_profit
                    partial_sold_cum = fraction_sold_target
                break  # Stop as soon as a match is found

        # If partial_sold_cum >= 1.0 => position fully closed
        if partial_sold_cum >= 1.0:
            net_profit = realized_profit_usd - entry_price * original_amount
            self.bot.log_exit_and_remove(
                idx, net_profit, realized_profit_usd, partial_sold_cum, max_price
            )
            return

        if partial_sold_cum == 0.0:
            # We haven't sold anything yet => fix stoploss = 70% of entry price
            stop_price = 0.70 * entry_price
        else:
            # 2) Trailing stop
            def get_trailing_stop_factor(mfactor: float):
                if mfactor < 30.0:
                    return 0.70 * mfactor  # 70% of mfactor
                else:
                    return mfactor

            stop_factor = get_trailing_stop_factor(max_factor)
            stop_price = entry_price * stop_factor

        leftover_frac = 1.0 - partial_sold_cum
        if leftover_frac > 0 and current_price < stop_price:
            if stoploss_count == 0:
                self.bot.positions_df.loc[idx, "stoploss_count"] = 1
                logging.info(
                    f"Stoploss breach detected for {token_addr[:6]}...: Price={current_price:.4f} < stop={stop_price:.4f}. "
                    "Waiting to confirm in next check."
                )
            elif stoploss_count == 1:
                self.bot.positions_df.loc[idx, "stoploss_count"] = (
                    2  # Optional: set to 2 or reset to 0 after selling
                )
                logging.info(
                    f"Stoploss confirmed for {token_addr[:6]}...: Price={current_price:.4f} < stop={stop_price:.4f} for second consecutive check."
                )

                tokens_leftover = original_amount * leftover_frac
                if tokens_leftover > 0:
                    logging.info(
                        f"{RED}Trailing stop triggered! Price={current_price:.4f} < stop={stop_price:.4f}. "
                        f"Selling leftover fraction={leftover_frac:.2%}.{RESET}"
                    )

                    self.bot.buy_cooldown_tokens.add(token_addr)  # Add to cooldown

                    logging.info(
                        f"Selling {tokens_leftover} tokens of {token_addr[:6]}..."
                    )
                    part_profit = self.bot.sell_token(
                        token_address=token_addr,
                        token_amount=tokens_leftover,
                        partial_sold_cumulative=float(1.0),
                        realized_pnl=float(realized_profit_usd),
                        stoploss_price=float(stop_price),
                        max_recorded_price=float(max_price),
                    )
                    if part_profit:
                        if realized_profit_usd is None:
                            realized_profit_usd = 0.0

                        realized_profit_usd += part_profit
                        partial_sold_cum = 1.0
                        logging.info(
                            f"{YELLOW}Position closed by trailing stop. Removing from DataFrame. (Token={token_addr[:6]}...){RESET}"
                        )
                        with self.bot.positions_lock:
                            self.bot.positions_df.loc[idx, "realized_profit_usd"] = (
                                realized_profit_usd
                            )
                            self.bot.positions_df.drop(idx, inplace=True)
                return
        else:
            # reset stoploss count
            if row.get("stoploss_count", 0) > 0:
                self.bot.positions_df.loc[idx, "stoploss_count"] = 0
                logging.info(
                    f"Stoploss breach reset for {token_addr[:6]} (price recovered above stoploss)."
                )

        # If not fully closed, just update partials and max_price
        with self.bot.positions_lock:
            if idx in self.bot.positions_df.index:
                self.bot.positions_df.loc[idx, "partial_sold_cumulative"] = (
                    partial_sold_cum
                )
                self.bot.positions_df.loc[idx, "realized_profit_usd"] = (
                    realized_profit_usd
                )
                self.bot.positions_df.loc[idx, "max_price"] = max_price


def main():
    bot = PiggyBackSniper(
        birdeye_api_key=Config.BIRDEYE_API_KEY,
        wallet_address=Config.SNIPER_WALLET_ADDRESS,
        private_key_base58=Config.SNIPER_KEY_BASE58,
        rpc_url=Config.RPC_URL,
        starting_usdc_balance=1000.0,  # Start with 1,000 USDC
        max_open_trades=5,
        usd_per_trade=5.0,
        mc_bounds=(50_000, 10_000_000),
        # scout_interval=(30, "minutes"), # 5 minutes
        scout_interval=(10, "seconds"),
        position_check_interval=10,  # 5 seconds
        paper_trading=False,  # Enable paper trading
        trade_opened_less_than_x_mins_ago=180,  # 3 hours
        random_sample_size=30,
        next_tradable_hours=24,
        sma_window_hours=24,
    )

    bot.run()


if __name__ == "__main__":
    main()
