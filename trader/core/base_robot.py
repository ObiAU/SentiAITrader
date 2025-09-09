import logging
import os
import pandas as pd
import schedule
import signal
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from typing import Tuple, Callable, List, Any, Optional, Literal

from solana.rpc.api import Client
from solders import keypair as Keypair
from ta.trend import ADXIndicator

warnings.simplefilter(action='ignore', category=FutureWarning)

from trader.config import Config
from trader.core.birdeye import BirdEyeClient, BirdEyeConfig
from trader.core.models import *
from trader.core.token_scout import scout_tokens, fetch_token_overview_and_metadata
from trader.database.log_utils import DatabaseLogger
from trader.database.supa_client import insert_row, update_row, execute_sql_query
from trader.sniper.analysis import analyze_for_good_tokens
from trader.sniper.jupiter import (
    swap_tokens,
    update_holdings_from_transaction,
    get_jupiter_quote,
    get_jupiter_swap_tx,
    sign_and_send_versioned_tx,
    get_jupiter_usd_price
)

# ---------------------------------------------------------
# ANSI color codes for selective colorful printing
# ---------------------------------------------------------
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
RESET = "\033[0m"

USDC_MINT_ADDRESS = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
WRAPPED_SOL_MINT_ADDRESS = "So11111111111111111111111111111111111111111"
WRAPPED_SOL_OHLCV_ADDRESS = "So11111111111111111111111111111111111111112"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class CooldownManager:
    def __init__(self, cooldown_period_seconds: int = 1800):
        self.cooldown_period = cooldown_period_seconds
        self.cooldowns = {}
        self.lock = threading.Lock()

    def add(self, key: str):
        """Add a key with an expiration time."""
        expiration_time = time.time() + self.cooldown_period
        with self.lock:
            self.cooldowns[key] = expiration_time

    def is_on_cooldown(self, key: str) -> bool:
        """Check if a key is on cooldown and clean up expired entries."""
        self.cleanup()
        with self.lock:
            return key in self.cooldowns

    def cleanup(self):
        """Remove expired keys."""
        current_time = time.time()
        with self.lock:
            self.cooldowns = {
                k: v for k, v in self.cooldowns.items() if v > current_time
            }

# Trades recording fix, upsert not working either, nonetype errors. Needs a manual run on Tuesday night to stress test everything.

class BaseBot:
    """
    A base class containing shared logic for Solana-based trading bots:
    - Holdings & positions DataFrame
    - Basic buy/sell methods
    - Threading & event loops
    - Logging, teardown/shutdown handling, etc.
    """

    # callable functions are: specific analysis, and sell strategy
    def __init__(
        self,
        strategy_function: Callable[..., List[Any]],
        wallet_address: str, # wallet address the bot will trade from
        birdeye_api_key: str,
        private_key_base58: str,
        analyze_function: Callable[..., List[Any]] = None,
        rpc_url: str = Config.RPC_URL,
        starting_usdc_balance: float = 1000.0,
        max_open_trades: int = 10,
        usd_per_trade: float = 5.0,
        scout_interval: Tuple[int, Literal['seconds', 'minutes', 'hours']] = (600, 'seconds'),
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
        creation_min_days_ago: int = 90,   # token must be created within these many days
        creation_max_hours_ago: int = 2,   # token must NOT be created too recently, e.g. last 2 hours
        trade_opened_less_than_x_mins_ago: Optional[int] = None, # trade must be opened within these many minutes
        random_sample_size: Optional[int] = None,
        retention_probability: float = 0.95,
        buy_cooldown_seconds: Optional[int] = 1800,
        sell_cooldown_seconds: Optional[int] = None,
        monitor_function: Optional[Callable[..., List[Any]]] = None,
        next_tradable_hours: Optional[int] = 48,
        check_interval: Optional[int] = 60,
        sma_window_hours: Optional[int] = 24,
    ):
        # funcs
        self.strategy_function = strategy_function
        self.analyze_function = analyze_function
        self.monitor_function = monitor_function
        # Store parameters
        self.wallet_address = wallet_address
        self.paper_trading = paper_trading
        self.private_key_base58 = private_key_base58
        self.rpc_url = rpc_url
        self.max_open_trades = max_open_trades
        self.usd_per_trade = usd_per_trade
        self.scout_interval, self.scout_interval_unit = scout_interval
        self.position_check_interval = position_check_interval
        self.analyze_potential_entries_interval = analyze_potential_entries_interval
        self.initial_balance = starting_usdc_balance
        self.next_tradable_hours = next_tradable_hours
        self.sma_window_hours = sma_window_hours
        self.potential_entries = {}
        self.holdings = {}
        self.buy_cooldown_tokens = CooldownManager(buy_cooldown_seconds)
        self.sell_cooldown_tokens = CooldownManager(sell_cooldown_seconds)

        self.market_trend = "none"

        self.trade_logger = DatabaseLogger(table_name, bot_type)
        self.bot_type = bot_type
        self.token_metadata_cache = {}


        # Scout configs
        self.mc_bounds = mc_bounds
        self.min_trades_in_30m = min_trades_in_30m
        self.max_drawdown_allowed = max_drawdown_allowed
        self.top10_holder_percent_limit = top10_holder_percent_limit
        self.creation_max_days_ago = creation_max_days_ago
        self.creation_min_days_ago = creation_min_days_ago
        self.creation_max_hours_ago = creation_max_hours_ago
        self.trade_opened_less_than_x_mins_ago = trade_opened_less_than_x_mins_ago
        self.random_sample_size = random_sample_size
        self.retention_probability = retention_probability

        # scout logic
        self.shutdown_event = False
        self.check_interval = check_interval # 1 minute
        self.next_scout_time = datetime.now(timezone.utc)

        # Solana Client
        self.solana_client = Client(rpc_url)

        # BirdEye Client
        bird_config = BirdEyeConfig(api_key=birdeye_api_key)
        self.bird_client = BirdEyeClient(config=bird_config)

        # Lock for thread safety (when reading/writing self.positions_df or self.holdings or potential po
        self.positions_lock = threading.RLock()  # For positions_df
        self.holdings_lock = threading.RLock()   # For holdings
        self.entries_lock = threading.RLock()    # For potential_entries


        balances = self.bird_client.fetch_balances(self.wallet_address)

        if not paper_trading:
            try:
                self.create_holdings(balances, negligible_value_threshold = 0.01)

                starting_usdc_balance = self.holdings.get(USDC_MINT_ADDRESS, {"balance": 0.0})["balance"]
                self.initial_balance = starting_usdc_balance

            except Exception as e:
                logging.error(f"Error fetching USDC balance: {e}")
                starting_usdc_balance = 0.0  # Default to 0 if fetching fails

        logging.info(f"{self.bot_type} executed")
        logging.info(f"{self.bot_type} executed")

        logging.info(f"{self.bot_type} BOT initialized with starting USDC balance: {starting_usdc_balance:.2f} USDC")
        logging.info(f"{MAGENTA}{self.bot_type} BOT initialized with starting USDC balance: {starting_usdc_balance:.2f} USDC{RESET}")

        try:
            self.holdings[USDC_MINT_ADDRESS] = {"balance": starting_usdc_balance or 0.0, "decimals": 6}
        except Exception as e:
            logging.error(f"Error adding USDC Mint to holdings. May already be present: {e}")

        # positions_df loading and qa
        self.positions_df = self.trade_logger.load_positions_from_db(self.wallet_address, balances)
        with self.positions_lock:
            self.positions_df = self.quality_check_positions(self.positions_df)

        logging.info(f"Current positions: {self.positions_df}")
        logging.info(f"Current positions: {self.positions_df}")

        self.shutdown_event = threading.Event()

        # Logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s"
        )
        logging.info(f"{bot_type} initialized"+". Paper Trading = %s", self.paper_trading)

    # --------------------------------------------------------------------
    # MARKET CONDITIONS UTILS
    # --------------------------------------------------------------------

    def update_market_trend(self, window: int = 14, adx_threshold: float = 23):
        """
        Fetch recent SOL candle data and update self.market_trend.
        Sets the trend to:
          - "negative" if DI⁻ > DI⁺ and ADX > threshold,
          - "positive" if DI⁺ > DI⁻ and ADX > threshold,
          - "none" otherwise.
        """
        now = int(time.time())
        time_from = now - window * 3600  # e.g., 14 hours of data using 1h candles

        try:
            candles = self.bird_client.fetch_candles(
                address=WRAPPED_SOL_OHLCV_ADDRESS,
                timeframe="30m",
                time_from=time_from,
                time_to=now
            )
        except Exception as e:
            logging.error(f"Error fetching SOL candles for market trend analysis: {e}")
            self.market_trend = "none"
            return

        if not candles:
            logging.warning("No SOL candle data available for market trend analysis.")
            self.market_trend = "none"
            return

        df = pd.DataFrame([c.model_dump() for c in candles]) # fetch candles gives list of Candle objects
        try:
            df['high'] = df['h'].astype(float)
            df['low'] = df['l'].astype(float)
            df['close'] = df['c'].astype(float)
        except Exception as e:
            logging.error(f"Error processing candle data for market trend analysis: {e}")
            self.market_trend = "none"
            return

        # Calculate ADX and its directional indices.
        adx_indicator = ADXIndicator(
            high=df['high'], 
            low=df['low'], 
            close=df['close'], 
            window=window
        )
        try:
            adx = adx_indicator.adx().iloc[-1]
            di_plus = adx_indicator.adx_pos().iloc[-1]
            di_minus = adx_indicator.adx_neg().iloc[-1]
        except Exception as e:
            logging.error(f"Error calculating ADX for market trend: {e}")
            self.market_trend = "none"
            return

        logging.info(f"Market Trend Analysis - ADX: {adx:.2f}, DI+: {di_plus:.2f}, DI-: {di_minus:.2f} at {datetime.now()}")

        if di_minus > di_plus and adx > adx_threshold:
            self.market_trend = "negative"
        elif di_plus > di_minus and adx > adx_threshold:
            self.market_trend = "positive"
        else:
            self.market_trend = "none"

    def is_market_bearish(self) -> bool:
        """
        for sniper if sma = bearish then pause for 12 hours
        for trader if sma = bearish then set next tradable to 1 week from now if it buys
        """
        pause_trading = False
        sma, current_price = self.check_market()
        if sma is None or current_price is None:
            logging.error("Could not fetch market conditions")
            return False

        self.update_market_trend()

        logging.info(f"Current trend: {self.market_trend}")

        # we could relax this condition to just "not being in a downtrend"
        if current_price < sma:
            pause_trading = True
            logging.info(
                f"Bearish market condition detected: current SOL price ({current_price:.4f}) "
                f"is below the {self.sma_window_hours}hr SMA ({sma:.4f})."
            )
        
        if self.market_trend == "negative":
            logging.info("Bearish market detected: market trend is negative, pause_trading flag set to True")
            pause_trading = True


        if pause_trading:
            if self.bot_type == "sniper":
                logging.info("Sniper detected bearish market: pausing trading for 12 hours.")
                return True

            elif self.bot_type == "sentitrader":
                logging.info("sentitrader detected bearish market: upserting tradable time for all positions to 1 week from now.")
                next_tradable_time = datetime.now(timezone.utc) + timedelta(weeks=1)
                with self.positions_lock:
                    for idx, row in self.positions_df.iterrows():
                        addr = row["token_address"]
                        position_id = self.get_or_create_position_id(addr)

                        self.positions_df.loc[idx, "next_tradable"] = next_tradable_time.isoformat()
                        try:
                            self.trade_logger.upsert_position(
                                position_id=position_id,
                                token_address=addr,
                                wallet_address=self.wallet_address,
                                next_tradable=next_tradable_time
                            )
                        except Exception as e:
                            logging.error(f"Failed to update next_tradable time in database for position {row['position_id']}: {e}")

                logging.info("sentitrader detected bearish market: "
                           "setting next tradable time for all positions to 1 week from now.")

                return True
        
        logging.info("Market conditions check passed, trading will continue")

        return False


    def check_market(self) -> Tuple[float, float]:
        """
        get sma of SOL over sma_window_hours
        """
        now = int(time.time())
        time_from = now - self.sma_window_hours * 3600
        try:
            candles = self.bird_client.fetch_candles(
                address=WRAPPED_SOL_OHLCV_ADDRESS,
                timeframe="1H",
                time_from=time_from,
                time_to=now
            )
        except Exception as e:
            logging.error(f"Error fetching SOL candles: {e}")
            return None, None

        if not candles:
            logging.warning("No candle data available for SOL market conditions.")
            return None, None

        close_prices = [c.c for c in candles]
        sma = sum(close_prices) / len(close_prices)

        try:
            current_price = self.bird_client.fetch_live_price(WRAPPED_SOL_OHLCV_ADDRESS)
        except Exception as e:
            logging.error(f"Error fetching current SOL price: {e}")
            return sma, None

        logging.info(
            f"Market Conditions - Current SOL Price: {current_price:.4f}, "
            f"{self.sma_window_hours}hr SMA: {sma:.4f}"
        )
        return sma, current_price

    # --------------------------------------------------------------------
    # HELPER METHODS
    # --------------------------------------------------------------------

    def is_token_on_exit_cooldown(self, token_address: str) -> bool:
        """
        True if token is not tradable yet
        """
        query = f"""
            SELECT next_tradable
            FROM positions
            WHERE token_address = '{token_address}'
            AND bot_executed = '{self.bot_type}'
            LIMIT 1;
        """
        result = execute_sql_query(query)
        if result and result[0].get("next_tradable"):
            next_tradable_str = result[0]["next_tradable"]
            try:
                next_tradable_time = datetime.fromisoformat(next_tradable_str)
            except Exception as e:
                logging.error(f"Error parsing next_tradable time for {token_address}: {e}")
                return False
            if next_tradable_time > datetime.now(timezone.utc):
                logging.info(f"Token {token_address} is on cooldown until {next_tradable_time}")
                return True
        return False


    def quality_check_positions(self, df):
        """
        qol cleaning for positions_df
        """
        logging.info("Performing QA on positions_df..")
        logging.info("Performing QA on positions_df..")
        expected_columns = [
            "position_id",
            "token_address",
            "ticker",
            "token_name",
            "entry_price",            # Price in USDC at time of purchase
            "entry_amount",           # How many tokens we bought
            "current_amount",         # Current amount of tokens held
            "partial_sold_cumulative",# Fraction of original tokens sold so far
            "realized_profit_usd",    # Total USD gained (or lost) from partial sells
            "max_price",              # Highest observed price so far (in USDC)
            "last_price",             # Last fetched price (in USDC) – won't store in DB
            "status",                 # "open" or "closed"
            "entry_time",
            "next_tradable",
            "stoploss_count"          # Count for stoploss confirmations
        ]

        if df is None or df.empty:
            df = pd.DataFrame(columns=expected_columns)
        else:
            for idx, row in df.iterrows():
                token_addr = row.get("token_address")
                try:
                    last_price = self.bird_client.fetch_live_price(token_addr)
                except Exception as e:
                    logging.error(f"Error fetching live price for token {token_addr}: {e}")
                    last_price = 0.0

                df.at[idx, "last_price"] = last_price

                if not row.get("entry_price") or row["entry_price"] == 0.0:
                    df.at[idx, "entry_price"] = last_price
                if not row.get("max_price") or row["max_price"] == 0.0:
                    df.at[idx, "max_price"] = last_price

        if "token_address" in df.columns and WRAPPED_SOL_MINT_ADDRESS in df["token_address"].values:
            logging.info("Removing Wrapped SOL from positions_df")
            df = df[df["token_address"] != WRAPPED_SOL_MINT_ADDRESS]

        for col in expected_columns:
            if col not in df.columns:
                default_value = 0 if col in ["entry_price", "entry_amount", "current_amount",
                                            "partial_sold_cumulative", "realized_profit_usd",
                                            "max_price", "last_price", "stoploss_count"] else ""
                df[col] = default_value
            else:
                if col in ["entry_price", "entry_amount", "current_amount",
                        "partial_sold_cumulative", "realized_profit_usd",
                        "max_price", "last_price", "stoploss_count"]:
                    df[col].fillna(0, inplace=True)

        return df


    def create_holdings(self, balances, negligible_value_threshold = 0.10):

        with self.holdings_lock:
            try:
                self.holdings = {}
                for token_info in balances:
                    token_decimals = token_info.decimals
                    token_balance = float(token_info.balance) / (10**token_decimals)
                    # token_ui_amount = token_info.uiAmount

                    token_value = token_balance * (token_info.priceUsd or 0.0)

                    if token_info.address not in [USDC_MINT_ADDRESS, WRAPPED_SOL_MINT_ADDRESS, WRAPPED_SOL_OHLCV_ADDRESS] and token_value < negligible_value_threshold:
                        logging.info(f"Filtered out token {token_info.symbol or 'N/A'} ({token_info.address}) "
                            f"with balance={token_balance:.6f}, value=${token_value:.2f}.")
                        logging.info(
                            f"Filtered out token {token_info.symbol or 'N/A'} ({token_info.address}) "
                            f"with balance={token_balance:.6f}, value=${token_value:.2f}."
                        )
                        continue

                    logging.info(f"[create_holdings] {token_info.address} => decimals from balances: {token_decimals}")
                    logging.info(f"[create_holdings] {token_info.address} => decimals from balances: {token_decimals}")

                    self.holdings[token_info.address] = {
                        "balance": token_balance,
                        "decimals": token_decimals,
                        "name": token_info.name,
                        "symbol": token_info.symbol,
                        "price_usd": token_info.priceUsd,
                        "last_update_time": 0,
                    }
                logging.info(f"Holdings created with {len(self.holdings)} tokens after filtering.")

            except Exception as e:
                logging.error(f"Error building holdings from balances: {e}")
                self.holdings = {}

    def consolidate_positions_db_onchain(self, sleep_time: int = None):

        if sleep_time > 0:
            logging.info(f"Sleeping for {sleep_time} seconds before consolidating positions.")
            logging.info(f"Sleeping for {sleep_time} seconds before consolidating positions.")
            time.sleep(sleep_time)

        try:
            balances = self.bird_client.fetch_balances(self.wallet_address)
            self.create_holdings(balances)

            logging.info(f"Holdings: {self.holdings}")
            logging.info(f"Holdings: {self.holdings}")

            df = self.trade_logger.load_positions_from_db(
                wallet_address=self.wallet_address,
                balances=balances,
                negligible_value_threshold=0.01
            )

            if df is None or df.empty:
                logging.info("No open positions found in DB after load_positions_from_db. Possibly everything is closed.")
                return

            if "last_price" not in df.columns:
                df["last_price"] = 0.0

            for idx, token in df.iterrows():

                last_price = self.bird_client.fetch_live_price(token["token_address"])

                df.at[idx, "last_price"] = last_price

                if not token["entry_price"] or token["entry_price"] == 0.0:
                    df.at[idx, "entry_price"] = last_price

            if "token_address" in df.columns and WRAPPED_SOL_MINT_ADDRESS in df["token_address"].values:
                logging.info(f"Removing Wrapped SOL from positions_df")
                df = df[df["token_address"] != WRAPPED_SOL_MINT_ADDRESS]

            try:
                df["max_price"] = df["max_price"].fillna(0.0)
                df["last_price"] = df["last_price"].fillna(0.0)
                df["realized_profit_usd"] = df["realized_profit_usd"].fillna(0.0)
            except Exception as e:
                logging.error(f"Failed to fill max_price with 0.0: {e}")

            with self.positions_lock:
                df = self.quality_check_positions(df)
                self.positions_df = df

            logging.info("Successfully consolidated DB positions with on-chain balances.")
        except Exception as e:
            logging.error(f"Error in consolidate_positions_db_onchain: {e}")

    def get_or_create_position_id(self, token_address: str) -> int:
        """
        Returns the existing position_id for a token, or creates one if none exists.
        """
        with self.positions_lock:
            if token_address in self.positions_df["token_address"].values and self.positions_df.loc[self.positions_df["token_address"] == token_address, "position_id"].iloc[0] is not None:
                logging.info(f"Position ID found in positions_df: {self.positions_df.loc[self.positions_df['token_address'] == token_address, 'position_id'].iloc[0]}")
                return self.positions_df.loc[self.positions_df["token_address"] == token_address, "position_id"].iloc[0]

        token_address = token_address.strip()
        query = f"""
            SELECT position_id FROM positions WHERE token_address = '{token_address}' and bot_executed = '{self.bot_type}' LIMIT 1
        """
        query = query.strip()
        result = execute_sql_query(query)

        if result:
            resultant = result[0].get("result")
            if resultant:
                position_id = resultant.get("position_id", None)
                logging.info(f"Position ID found in DB: {position_id}")
                with self.positions_lock:
                    new_row = {"token_address": token_address, "position_id": position_id}
                    self.positions_df.loc[len(self.positions_df)] = new_row # append deprecated in pandas > 2.0 (not sure how we missed this)
                return position_id

        # If it's not in DB, create a new row in "positions" w just token address and status
        insert_result = insert_row("positions", {"token_address": token_address, "status": "open", "bot_executed": self.bot_type})

        if not insert_result:
            raise Exception(f"Failed to insert row for token {token_address}")

        # we get position_id back / inserted row back from db insert
        new_position_id = insert_result[0]["position_id"]
        with self.positions_lock:
            new_row = {"token_address": token_address, "position_id": new_position_id}
            # self.positions_df = self.positions_df.append(new_row, ignore_index=True)
            self.positions_df.loc[len(self.positions_df)] = new_row

        return new_position_id

    def fetch_token_price_in_usdc(self, token_mint: str) -> float:
        """
        Fetch the current token price in USDC from BirdEye.
        If it fails, fallback to 0.0 to indicate an error.
        """
        try:
            return self.bird_client.fetch_live_price(token_mint)
        except Exception as e:
            logging.error(f"Error fetching price for {token_mint}: {e}")
            return 0.0

    def get_usdc_balance(self) -> float:
        """
        Convenience method to fetch how much USDC we currently hold.
        Returns the numeric balance from the holdings dictionary.
        """
        with self.holdings_lock:
            usdc_info = self.holdings.get(USDC_MINT_ADDRESS, {"balance": 0.0})
            return usdc_info.get("balance", 0.0)


    # --------------------------------------------------------------------
    # Transaction Methods
    # --------------------------------------------------------------------


    def buy_token(self, token_address: str, usdc_amount: float, ticker: str = "", token_name: str = ""):
        """
        Buys usdc_amount worth of token_address using Jupiter swap from USDC,
        unless paper_trading=True (in which case we just simulate).

        Returns a dict with the trade details (or None if buy failed).
        """
        if self.bot_type == "sniper":
            if self.is_token_on_exit_cooldown(token_address):
                logging.info(f"{RED}[BUY] Token {token_address} for {self.bot_type} is on cooldown. Skipping.")
                return None

        current_usdc = self.get_usdc_balance()

        # qol decimals override
        try:
            true_decimals = self.bird_client.fetch_token_overview(token_address).decimals
            with self.holdings_lock:
                try:
                    if token_address not in self.holdings:
                        self.holdings[token_address] = {"balance": 0.0}  # Initialize if not present
                    self.holdings[token_address]["decimals"] = true_decimals

                except Exception as e:
                    logging.error(f"Error updating decimals for {token_address}: {e}")
                    logging.error(f"Error updating decimals for {token_address}: {e}")

            logging.info(f"[buy_token] {token_address} => decimals from overview: {true_decimals}")
            logging.info(f"[buy_token] {token_address} => decimals from overview: {true_decimals}")
        except Exception as e:
            logging.error(f"Error fetching token overview for {token_address}: {e}")
            logging.error(f"Error fetching token overview for {token_address}: {e}")

        try:
            if self.buy_cooldown_tokens.is_on_cooldown(token_address):
                logging.info(f"{RED}[BUY] Token {token_address} is on cooldown. Skipping.")
                return None
        except Exception as e:
            logging.error(f"Error checking cooldown for {token_address}: {e}")
            logging.error(f"Error checking cooldown for {token_address}: {e}")

        if usdc_amount > current_usdc:
            logging.info(
                f"{CYAN}Insufficient USDC. Balance: {current_usdc:.2f}, Needed: {usdc_amount:.2f}"
            )
            return None

        if not self.paper_trading:
            # Actual Jupiter call
            try:
                logging.info(f"Buying ${usdc_amount:.2f} of {token_address} ...")
                tx_sig, status, exact_out_amount, block_time = swap_tokens(
                    private_key_base58=self.private_key_base58,
                    rpc_url=self.rpc_url,
                    input_mint=USDC_MINT_ADDRESS,  # USDC
                    input_mint_decimals=6,
                    # output_mint_decimals=true_decimals,
                    output_mint=token_address,
                    amount_in=usdc_amount,
                    slippage_bps=250,
                    max_retries=3,
                    retry_delay=10,
                    bird_client=self.bird_client,
                    local_holdings=self.holdings,
                    holdings_lock=self.holdings_lock,
                    wallet_address=self.wallet_address
                )

                if status != "finalized" or not tx_sig:
                    logging.error("Swap ultimately failed even after BirdEye fallback.")
                    return 0.0

                entry_price = usdc_amount / exact_out_amount
                max_price = entry_price
                block_time = datetime.fromtimestamp(block_time, timezone.utc)

                if self.bot_type == "sentitrader":
                    min_hold_delta = timedelta(hours=self.next_tradable_hours)
                    next_tradable_time = block_time + min_hold_delta
                else:
                    next_tradable_time = None

                logging.info(f"Bought {exact_out_amount} (${usdc_amount}) of {token_address}: tx({tx_sig})")
                logging.info(f"{GREEN}Bought {exact_out_amount} (${usdc_amount}) of {token_address}: tx({tx_sig}){RESET}")

                # doesnt matter if within buy, as will inevitable call for new tokens
                if token_address in self.token_metadata_cache:
                    metadata = self.token_metadata_cache[token_address]

                else:
                    metadata = fetch_token_overview_and_metadata(token_address)
                    self.token_metadata_cache[token_address] = metadata

                ticker = metadata.get("ticker", "")
                token_name = metadata.get("name", "")

                position_id = self.get_or_create_position_id(token_address)

                try:
                    self.trade_logger.record_trade(
                        position_id=position_id,
                        timestamp=block_time,
                        token_address=token_address,
                        ticker = ticker,
                        token_name = token_name,
                        tx_signature=tx_sig,
                        entry_exit_price=entry_price,
                        amount=exact_out_amount,
                        buy_sell="buy",
                        wallet_address = self.wallet_address
                        )

                    # rare case where we buy the same token at a higher price
                    if token_address in self.positions_df["token_address"].values and self.positions_df.loc[self.positions_df["token_address"] == token_address, "entry_price"].iloc[0] < entry_price:
                        entry_price = None

                    self.trade_logger.upsert_position(
                        position_id=position_id,
                        entry_time= block_time,
                        entry_amount=exact_out_amount,
                        entry_price=entry_price,
                        partial_sold_cumulative=0.0,
                        last_trade_time=block_time,
                        token_address=token_address,
                        ticker_symbol=ticker,
                        token_name=token_name,
                        next_tradable=next_tradable_time,
                        blockchain="solana",
                        amount_holding=self.holdings[token_address].get("balance", None),  # total holdings now
                        amount_sold=None,
                        realized_pnl=None,
                        trade_status="open",
                        type="market",
                        wallet_address = self.wallet_address,
                        stoploss_price=None,
                        max_recorded_price=max_price,
                    )

                except Exception as e:
                    logging.error(f"{RED}Error recording trade of {token_address} at {block_time}")

                # Deduct USDC from holdings
                with self.holdings_lock:
                    if USDC_MINT_ADDRESS in self.holdings:
                        self.holdings[USDC_MINT_ADDRESS]["balance"] -= usdc_amount
                    else:
                        logging.warning(f"USDC balance not found in holdings. Skipping deduction.")

                token_decimals = self.holdings.get(token_address, {}).get("decimals", 0)

                try:
                    self.consolidate_positions_db_onchain(sleep_time=10) # sleep for 10seconds before consolidation
                except Exception as e:
                    logging.error(f"Error consolidating positions: {e}")
                    logging.error(f"Error consolidating positions: {e}")

                return {
                    "token_address": token_address,
                    "ticker": ticker,
                    "token_name": token_name,
                    "entry_price": entry_price,
                    "entry_amount": exact_out_amount,
                    "current_amount": exact_out_amount,
                    "tx_signature": tx_sig,
                }

            except Exception as e:
                logging.error(f"Error buying token {token_address}: {e}")
                return None

        else:
            # log Paper trading with colour of magenta
            logging.info(f"{CYAN}[PAPER] Buying ${usdc_amount:.2f} of {token_address} {RESET}")
            entry_price = self.fetch_token_price_in_usdc(token_address)
            token_decimals = self.holdings.get(token_address, {}).get("decimals", 0)
            if entry_price <= 0:
                logging.warning("[PAPER] Entry price <= 0, setting to 0.")
                entry_price = 0.0

            token_amount = 0.0
            if entry_price > 0:
                token_amount = usdc_amount / entry_price


            with self.holdings_lock:
                # Deduct USDC from holdings
                if USDC_MINT_ADDRESS in self.holdings:
                    self.holdings[USDC_MINT_ADDRESS]["balance"] -= usdc_amount
                else:
                    logging.warning(f"USDC balance not found in holdings. Skipping deduction.")

            # Update holdings for the purchased token
            if token_address in self.holdings:
                self.holdings[token_address]["balance"] += token_amount
            else:
                # Add new token entry if not already in holdings
                self.holdings[token_address] = {
                    "balance": token_amount,
                    "decimals": token_decimals,  # This assumes token_decimals is defined
                }

            tx_sig = f"PAPER_BUY_{time.time()}"
            return {
                "token_address": token_address,
                "ticker": ticker,
                "token_name": token_name,
                "entry_price": entry_price,
                "entry_amount": token_amount,
                "current_amount": token_amount,
                "tx_signature": tx_sig,
            }

    def log_exit_and_remove(self, idx, net_profit, realized_profit_usd, partial_sold_cum, max_price, reason = None):
        color = GREEN if net_profit > 0 else RED
        logging.info(
            f"{YELLOW}Position fully closed, final profit: {color}${net_profit:.2f}{RESET}"
            f"Reason: {reason}"
        )
        with self.positions_lock:
            # Update final realized profit
            self.positions_df.loc[idx, "realized_profit_usd"] = realized_profit_usd
            self.positions_df.loc[idx, "partial_sold_cumulative"] = partial_sold_cum
            self.positions_df.loc[idx, "max_price"] = max_price

            if self.bot_type == "sniper":
                next_tradable_time = datetime.now(timezone.utc) + timedelta(hours=self.next_tradable_hours)
            else:
                next_tradable_time = None

            try:
                self.trade_logger.upsert_position(
                    position_id = self.positions_df.loc[idx, "position_id"],
                    token_address = self.positions_df.loc[idx, "token_address"],
                    wallet_address = self.wallet_address,
                    trade_status = "closed",
                    realized_pnl = realized_profit_usd,  # final net
                    max_recorded_price = max_price,
                    partial_sold_cumulative=partial_sold_cum,
                    next_tradable=next_tradable_time
                )

            except Exception as e:
                logging.error(f"Failed to upsert positions at full exit step. Error {e}")
                logging.error(f"Failed to upsert positions at full exit step. Error {e}")

            self.positions_df.drop(idx, inplace=True)


    def sell_token(
                self,
                token_address: str,
                token_amount: float,
                partial_sold_cumulative: float = 0.0,
                realized_pnl: float = 0.0,
                stoploss_price: Optional[float] = None,
                max_recorded_price: Optional[float] = None
            ):
        """
        Sells a partial amount of token_address into USDC.
        Returns the total USD realized from the sale.

        Updates holdings and positions accordingly.
        """
        DUST_THRESHOLD = 0.00001 # maybe have dust threshold as value rather than token amount

        onchain_balance = self.bird_client.get_token_balance(self.wallet_address, token_address)
        token_amount = min(token_amount, onchain_balance) # in case trying to sell more than we have

        if token_amount < DUST_THRESHOLD:
            logging.info(f"Skipping dust sell: {token_amount} < {DUST_THRESHOLD}")
            return 0.0

        if self.sell_cooldown_tokens and self.sell_cooldown_tokens.is_on_cooldown(token_address):
            logging.info(f"Token {token_address} is still on cooldown. Skipping sell.")
            logging.info(f"Token {token_address} is still on cooldown. Skipping sell.")
            return 0.0

        # enforce next_tradable_time on sell -- will only apply to sentitrader
        if self.bot_type == "sentitrader":
            with self.positions_lock:
                next_tradable_time = self.positions_df.loc[self.positions_df["token_address"] == token_address, "next_tradable"].iloc[0]
                if isinstance(next_tradable_time, datetime):
                    next_tradable_time = next_tradable_time.replace(tzinfo=timezone.utc)
                if next_tradable_time and datetime.now(timezone.utc) < next_tradable_time:
                    logging.info(f"Token {token_address} is not yet tradable. Skipping sell.")
                    logging.info(f"Token {token_address} is not yet tradable. Skipping sell.")
                    return 0.0

        # qol decimals override
        true_decimals = self.bird_client.fetch_token_overview(token_address).decimals
        self.holdings[token_address]["decimals"] = true_decimals # test image update

        with self.holdings_lock:
            token_in_info = self.holdings.get(token_address, {})
            token_in_decimals = token_in_info.get("decimals", 0)
            current_token_in_balance = token_in_info.get("balance", 0.0)

        if token_amount > current_token_in_balance:
            token_amount = current_token_in_balance

        if not self.paper_trading:
            try:
                logging.info(f"Selling {token_amount:.6f} tokens of {token_address}...")
                logging.info(f"Selling {token_amount:.6f} tokens of {token_address}...")

                tx_sig, status, estimated_out_amount, block_time = swap_tokens(
                    private_key_base58=self.private_key_base58,
                    rpc_url=self.rpc_url,
                    input_mint=token_address,
                    input_mint_decimals=token_in_decimals,
                    output_mint_decimals=6,  # USDC has 6 decimals
                    output_mint=USDC_MINT_ADDRESS,
                    amount_in=token_amount,  # Pass as a human-readable float
                    slippage_bps=500, # 5% slippage tolerance which is high to avoid failures
                    max_retries=3,
                    retry_delay=10,
                    bird_client=self.bird_client,
                    local_holdings=self.holdings,
                    holdings_lock=self.holdings_lock,
                    wallet_address=self.wallet_address
                    )

                if status != "finalized" or not tx_sig:
                    logging.error("Swap ultimately failed even after BirdEye fallback.")
                    logging.error("Swap ultimately failed even after BirdEye fallback.")
                    return 0.0

                execution_price = 0.0
                block_time = datetime.fromtimestamp(block_time, timezone.utc)

                token_balance = max(0.0, current_token_in_balance - token_amount)
                trade_status = "closed" if token_balance <= 0.0 else "open"

                if trade_status == "closed" and self.bot_type == "sniper":
                    next_tradable_time = datetime.now(timezone.utc) + timedelta(hours=self.next_tradable_hours)
                else:
                    next_tradable_time = None

                if token_amount > 0:
                    execution_price = estimated_out_amount / token_amount

                if isinstance(tx_sig, str) and "error" in tx_sig.lower():
                    logging.error(f"Swap failed with error: {tx_sig}")
                    logging.error(f"Swap failed with error: {tx_sig}")
                    return 0.0


                with self.positions_lock:
                    # Update positions DataFrame
                    self.positions_df.loc[
                        self.positions_df["token_address"] == token_address, "current_amount"
                    ] -= token_amount

                try:
                    position_id = self.positions_df.loc[self.positions_df["token_address"] == token_address, "position_id"].iloc[0]
                    token_name = self.positions_df.loc[self.positions_df["token_address"] == token_address, "token_name"].iloc[0]
                    ticker = self.positions_df.loc[self.positions_df["token_address"] == token_address, "ticker"].iloc[0]

                    self.trade_logger.record_trade(
                        position_id=position_id,
                        timestamp=block_time,
                        token_address=token_address,
                        ticker = ticker,
                        token_name = token_name,
                        tx_signature=tx_sig,
                        entry_exit_price=execution_price,
                        amount=estimated_out_amount,
                        buy_sell="sell",
                        wallet_address = self.wallet_address
                        )

                    self.trade_logger.upsert_position(
                        position_id=position_id,
                        entry_time = None,
                        entry_amount = None,
                        entry_price = None,
                        partial_sold_cumulative=partial_sold_cumulative,
                        last_trade_time = block_time,
                        token_address=token_address,
                        ticker_symbol=ticker,
                        token_name=token_name,
                        blockchain="solana",
                        amount_holding=self.holdings[token_address]["balance"],  # total holdings now
                        amount_sold=token_amount,
                        realized_pnl=realized_pnl + estimated_out_amount,
                        trade_status=trade_status,
                        type="market",
                        wallet_address = self.wallet_address,
                        stoploss_price=stoploss_price,
                        max_recorded_price=max_recorded_price,
                        next_tradable=next_tradable_time
                    )

                except Exception as e:
                    logging.error(f"{RED}Error recording trade of {token_address} at {block_time}")

                logging.info(f"Sold {token_amount:.6f} tokens of {token_name}. Realized profit: {realized_pnl + estimated_out_amount}")
                logging.info(f"Sold {token_amount:.6f} tokens of {token_name}. Realized profit: {realized_pnl + estimated_out_amount}")


                try:
                    self.consolidate_positions_db_onchain(sleep_time=10) # sleep for 10seconds before consolidation
                except Exception as e:
                    logging.error(f"Error consolidating positions: {e}")
                    logging.error(f"Error consolidating positions: {e}")

                if not isinstance(estimated_out_amount, (float, int)) or not estimated_out_amount:
                    logging.error(f"Estimated out amount is not a valid number: {estimated_out_amount}")
                    logging.error(f"Estimated out amount is not a valid number: {estimated_out_amount}")
                    estimated_out_amount = 0.0

                return estimated_out_amount # best we can do is use the estimated amount here.

            # returning 0.0 as fallback
            except Exception as e:
                logging.error(f"Error selling token {token_address}: {e}")
                logging.error(f"Error selling token {token_address}: {e}")
                return 0.0
        else:
            # Paper trading
            logging.info(f"[PAPER] Selling {token_amount:.6f} tokens of {token_address}...")
            sell_price = self.fetch_token_price_in_usdc(token_address)
            total_sale_usdc = token_amount * sell_price

            with self.holdings_lock, self.positions_lock:
                self.holdings[token_address]["balance"] = max(0.0, current_token_in_balance - token_amount)
                self.holdings[USDC_MINT_ADDRESS]["balance"] += total_sale_usdc

                # Update positions DataFrame
                self.positions_df.loc[
                    self.positions_df["token_address"] == token_address, "current_amount"
                ] -= token_amount

            return total_sale_usdc

    # ---------------------------------------------------------
    # TOP-LEVEL BOT METHODS
    # ---------------------------------------------------------


    def apply_strategy(self, idx: int, current_price: float):
        if self.strategy_function:
            self.strategy_function(idx, current_price)
        else:
            pass

    def add_new_position(self, trade_info: dict):
        """
        Adds a newly opened position to the DataFrame.
        """
        if not trade_info:
            return

        new_entry = {
            "token_address": trade_info["token_address"],
            "ticker": trade_info["ticker"],
            "token_name": trade_info["token_name"],
            "entry_price": trade_info["entry_price"],
            "entry_amount": trade_info["entry_amount"],
            "current_amount": trade_info["current_amount"],
            "partial_sold_cumulative": 0.0,
            "realized_profit_usd": 0.0,
            "max_price": trade_info["entry_price"],
            "last_price": trade_info["entry_price"],
            "status": "open",
            "entry_time": time.time(),
            "stoploss_count": 0,
            "next_tradable": None
        }
        with self.positions_lock:
            self.positions_df = pd.concat(
                [self.positions_df, pd.DataFrame([new_entry])],
                ignore_index=True
            )

    def check_positions(self):
        """
        Parallelized version of check_positions that uses a ThreadPoolExecutor.
        Each token is processed concurrently.
        """
        with self.positions_lock:
            if self.positions_df.empty:
                return
            open_idxs = self.positions_df[self.positions_df["status"] == "open"].index.tolist()

        def process_token(idx):
            """
            Function to process each token: fetch price, update `last_price`, and apply strategy.
            """
            with self.positions_lock:
                if idx not in self.positions_df.index:
                    return  # Might have been removed mid-loop
                row = self.positions_df.loc[idx]
                if row["status"] != "open":
                    return
                token_address = row["token_address"]

            # Fetch the current price
            if token_address not in self.holdings:
                return
            output_mint_decimals = self.holdings.get(token_address, {}).get("decimals", 0)
            time.sleep(0.5)  # Sleep to avoid rate limiting

            current_price = get_jupiter_usd_price(output_mint=token_address, output_mint_decimals=output_mint_decimals)
            logging.info(f"Current price of {token_address}: {current_price:.4f}")
            if not isinstance(current_price, (float, int)) or not current_price:
                return

            if current_price <= 0:
                return

            # Update `last_price` and apply the strategy
            position_id = self.get_or_create_position_id(token_address)
            max_recorded_price = self.trade_logger.get_max_recorded_price(token_address, position_id)

            with self.positions_lock:
                if idx in self.positions_df.index:
                    self.positions_df.loc[idx, "last_price"] = current_price
                    self.positions_df.loc[idx, "max_price"] = max_recorded_price

                    if max_recorded_price < current_price:
                        logging.info(f"Max price updated for {token_address}: {current_price:.4f}")
                        logging.info(f"Max price updated for {token_address}: {current_price:.4f}")
                        self.positions_df.loc[idx, "max_price"] = current_price

                        try:
                            self.trade_logger.upsert_position(
                                position_id=position_id,
                                wallet_address=self.wallet_address,
                                token_address=token_address,
                                max_recorded_price=current_price
                            )
                            logging.info(f"Upserted new max_price for position_id {position_id}: {current_price}")
                        except Exception as e:
                            logging.error(f"Failed to upsert max_price for position_id {position_id}: {e}")


            self.apply_strategy(idx, current_price)

        # Submit tasks to the executor
        futures = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            for idx in open_idxs:
                futures.append(executor.submit(process_token, idx))

        # Wait for all futures to complete (optional)
        for future in futures:
            try:
                future.result()  # This will raise exceptions if any occurred in the threads
            except Exception as e:
                logging.error(f"Error processing token in check_positions: {e}")


    def scout_and_buy(self):
        """
        Calls scout_tokens() to find new tokens, then attempts to buy
        self.usd_per_trade of each, if capacity allows.
        """
        logging.info(f"{CYAN}Token Scout searching for new tokens...{RESET}")
        candidate_tokens = scout_tokens(
            mc_bounds=self.mc_bounds,
            min_trades_in_30m=self.min_trades_in_30m,
            max_drawdown_allowed=self.max_drawdown_allowed,
            top10_holder_percent_limit=self.top10_holder_percent_limit,
            creation_max_days_ago=self.creation_max_days_ago,
            creation_min_days_ago=self.creation_min_days_ago,
            creation_max_hours_ago=self.creation_max_hours_ago,
            trade_opened_less_than_x_mins_ago=self.trade_opened_less_than_x_mins_ago,
            random_sample_size=self.random_sample_size,
            retention_probability=self.retention_probability,
            bot_type = self.bot_type
        )

        logging.info(f"{YELLOW}scout_tokens returned {len(candidate_tokens)} candidates{RESET}")
        if self.bot_type == "sentitrader":
            tokens_with_analysis = self.analyze_strategy(candidate_tokens)

        else:
            tokens_with_analysis = self.analyze_and_buy(candidate_tokens)

        if not tokens_with_analysis:
            logging.info("No valid tokens. Returning...")
            return


        def process_token(token):
            """
            Processes a single token: checks conditions and buys if valid.
            """

            # Check position and count open positions while we have the lock
            with self.positions_lock:
                already_have = any(self.positions_df["token_address"] == token.address)
                current_open = len(self.positions_df[self.positions_df["status"] == "open"])
                if current_open >= self.max_open_trades:
                    logging.info(f"Max open trades ({self.max_open_trades}) reached. Skipping.")
                    return
                if already_have:
                    logging.info(f"Already have a position in {token.address}. Skipping.")
                    return

            if not token.open_new_position and self.bot_type == "sniper":
                # Monitor for a better entry
                with self.positions_lock:
                    logging.info(f"Monitoring {token.address} for a better entry.")
                    self.potential_entries[token.address] = token
                return

            # Prepare to buy the token
            with self.holdings_lock:
                if self.holdings.get(token.address, {}).get("balance", 0.0) > 0:
                    self.holdings[token.address] = {"balance": 0.0, "decimals": token.decimals, "trader": token.trader}


            try:
            # Perform the buy operation outside the lock
                logging.info(f"Buying {token.address} with {self.usd_per_trade} USD...")
                trade_result = self.buy_token(token.address, self.usd_per_trade)
            except Exception as e:
                logging.error(f"Error buying token {token.address}: {e}")
                return

            if trade_result:
                self.add_new_position(trade_result)
                with self.entries_lock:

                    if self.bot_type == "sniper":
                        self.potential_entries.pop(token.address, None)
            else:
                logging.info(f"Failed to open position for {token.address}.")

        # Parallelize token processing
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(process_token, token) for token in tokens_with_analysis]

            for future in futures:
                future.result()
                # try:
                #     future.result()
                # except Exception as e:
                #     logging.error(f"Error processing token in analyze_and_buy: {e}")


    def analyze_strategy(self, candidate_tokens):
        if self.analyze_function:
            return self.analyze_function(candidate_tokens)
        return []

    def analyze_and_buy(self, candidate_tokens: list[TokenMetadata]):
        """
        Analyze candidate tokens and buy them in parallel.
        """
        if not candidate_tokens:
            return

        tokens_with_analysis = analyze_for_good_tokens(self.bird_client, candidate_tokens, self.trade_opened_less_than_x_mins_ago)

        return tokens_with_analysis


    # --------------------------------------------------------------------
    # BALANCE & PNL
    # --------------------------------------------------------------------
    def compute_total_pnl_positions(self) -> float:
        """
        Computes the total PnL across all *open positions* in positions_df:
        realized + unrealized, in USDC.
        (Note that closed positions are removed from the DF.)
        """
        total_pnl = 0.0
        with self.positions_lock:
            if self.positions_df.empty:
                return 0.0
            for _, row in self.positions_df.iterrows():
                entry_price = row["entry_price"]
                entry_amount = row["entry_amount"]
                last_price = row["last_price"]
                realized_profit = row["realized_profit_usd"]
                current_amount = row["current_amount"]

                # Unrealized PnL based on current_amount
                unrealized_pnl = current_amount * (last_price )
                total_pnl += realized_profit + unrealized_pnl - (entry_price * entry_amount)
        return total_pnl

    def compute_total_holdings_value(self) -> float:
        """
        Computes the total value of all holdings in USDC:
        - USDC directly in self.holdings[USDC_MINT_ADDRESS]
        - plus each token * its current price in USDC.
        """

        total_value = 0.0

        for item in self.holdings:
            amount = self.holdings[item].get("balance")

            if item == USDC_MINT_ADDRESS:
                    total_value += amount
                    continue

            price_in_usdc = self.fetch_token_price_in_usdc(item)

            if amount and price_in_usdc:
                total_value += (amount * price_in_usdc)

        return total_value


    # ---------------------------------------------------------
    # THREADING: SCOUTING & POSITION CHECKING
    # ---------------------------------------------------------
    def scout_task(self):
        """
        variant scouting job
        """
        def job():

            # alt pause logic
            now = datetime.now(timezone.utc)
            if now < self.next_scout_time:
                logging.info(f"Scout job paused until {self.next_scout_time.isoformat()}. Skipping this cycle.")
                return
            
            try:
                self.buy_cooldown_tokens.cleanup()
            except Exception as e:
                logging.exception(f"Unhandled exception in buy_cooldown_tokens.cleanup: {e}")

            is_bearish = self.is_market_bearish()
            if self.bot_type == "sniper" and is_bearish:
                self.next_scout_time = now + timedelta(hours=12)
                logging.info("Sniper bot: Market conditions unfavorable; skipping scouting this cycle.")
                return  # only pause if bot is sniper


            with self.positions_lock:
                current_open = len(self.positions_df[self.positions_df["status"] == "open"])
            if current_open < self.max_open_trades:
                    logging.info(f"Initiating scouting and buying...")
                    logging.info(f"Initiating scouting and buying...")
                    self.scout_and_buy()

            else:
                logging.info("Max open trades reached. Skipping scouting and buying.")
                logging.info("Max open trades reached. Skipping scouting and buying.")

        logging.info(f"Scout thread restarting with an interval of {self.scout_interval} {self.scout_interval_unit}.")
        logging.info(f"Scout thread restarting with an interval of {self.scout_interval} {self.scout_interval_unit}.")

        if self.scout_interval_unit == "seconds":
            schedule.every(self.scout_interval).seconds.do(job)
            sleep_time = self.scout_interval

        elif self.scout_interval_unit == "minutes":
            schedule.every(self.scout_interval).minutes.do(job)
            sleep_time = self.scout_interval * 60

        elif self.scout_interval_unit == "hours":
            schedule.every(self.scout_interval).hours.do(job)
            sleep_time = self.scout_interval * 3600

        else:
            raise ValueError("Invalid unit! Use 'seconds', 'minutes', or 'hours'.")

        while True:
            schedule.run_pending()
            logging.info(f"Scout thread sleeping for {sleep_time:.0f} seconds...")
            logging.info(f"Scout thread sleeping for {sleep_time:.0f} seconds...")
            time.sleep(sleep_time)



    def position_check_loop(self):
        """
        Runs in a dedicated thread: calls check_positions() every self.position_check_interval seconds.
        """
        # start a timer, onceit hits 1 minutes, set the flag to print positions = true

        timer = time.time()

        while True:
            try:
                self.check_positions()
            except Exception as e:
                logging.exception("Unhandled exception in position_check_loop:")

            if time.time() - timer > 60:
                # self.print_positions_summary()
                timer = time.time()
            time.sleep(self.position_check_interval)

    def analyze_potential_entries_loop(self):
        """
        Runs in a dedicated thread: analyzes potential entries
        """
        # start a timer, onceit hits 1 minutes, set the flag to print positions = true

        while True:
            try:
                # Discard stuff which the trader entered over an hour ago
                for token in list(self.potential_entries.values()):  # Convert to list to safely iterate while modifying the dict
                    time_since_tx = time.time() - token.tx_time.timestamp()

                    if time_since_tx > self.trade_opened_less_than_x_mins_ago*60:  # 1 hour
                        logging.info(f"Discarding {token.address} as it was entered over an hour ago.")
                        self.potential_entries.pop(token.address, None)


                tokens = list(self.potential_entries.values())

                self.analyze_and_buy(tokens)
            except Exception as e:
                logging.exception("Unhandled exception in position_check_loop:")

            time.sleep(self.analyze_potential_entries_interval)

    def handle_signal(self, signum, frame):
        """
        Signal handler that initiates shutdown.
        """
        logging.info(f"{YELLOW}Received signal {signum}. Initiating shutdown...{RESET}")
        self.shutdown_event.set()
        self.shutdown()

    def register_signal_handlers(self):
        """
        Registers signal handlers for graceful shutdown.
        """
        signal.signal(signal.SIGINT, self.handle_signal)   # Ctrl+C
        signal.signal(signal.SIGTERM, self.handle_signal)  # Termination signal


    def shutdown(self):
        """
        Sells all non-USDC open positions back to USDC in parallel and then exits the program.
        """
        logging.info(f"{RED}Shutdown initiated. Selling all open positions...{RESET}")

        with self.positions_lock:
            open_positions = self.positions_df[self.positions_df["status"] == "open"].copy()

        def sell_position(idx, row):
            """
            Sells a single position (row in the positions_df).
            Returns (idx, realized_profit) so we can update
            the DataFrame after the concurrent operations complete.
            """
            token_address = row["token_address"]
            current_amount = row["current_amount"]

            # Skip if token is USDC or amount <= 0
            if token_address == USDC_MINT_ADDRESS or current_amount <= 0:
                return idx, 0.0

            # Fetch current price
            current_price = self.fetch_token_price_in_usdc(token_address)
            if current_price <= 0:
                logging.warning(f"Cannot fetch price for {token_address}. Skipping sell.")
                return idx, 0.0

            # Sell all tokens
            realized_profit = self.sell_token(
                token_address=token_address,
                token_amount=current_amount,
            )

            logging.info(
                f"{GREEN}Sold {current_amount:.6f} tokens of {token_address[:6]}... => "
                f"Realized PnL: ${realized_profit:.2f}{RESET}"
            )
            return idx, realized_profit

        # Submit concurrent selling tasks
        futures = {}
        with ThreadPoolExecutor(max_workers=10) as executor:
            for idx, row in open_positions.iterrows():
                futures[executor.submit(sell_position, idx, row)] = idx

            # Collect results and update DataFrame accordingly
            for future in as_completed(futures):
                idx, realized_profit = future.result()
                if idx in self.positions_df.index:
                    with self.positions_lock:
                        self.positions_df.loc[idx, "realized_profit_usd"] += realized_profit
                        self.positions_df.loc[idx, "current_amount"] = 0.0
                        self.positions_df.loc[idx, "partial_sold_cumulative"] = 1.0
                        self.positions_df.loc[idx, "status"] = "closed"

        logging.info(f"{YELLOW}All open positions have been closed.{RESET}")
        # self.print_positions_summary()
        logging.info(f"{GREEN}Bot Shutdown!{RESET}")
        sys.exit(0)


    def run(self):
        """
        Launch two threads in parallel:
         1) The scout thread (scout_loop)
         2) The position check thread (position_check_loop)

        The main thread simply keeps the program alive.
        """
        self.register_signal_handlers()

        logging.info(f"{GREEN}=== BOT STARTED ==={RESET}")
        start_balance = self.get_usdc_balance()
        logging.info(f"{GREEN}Starting USDC Balance: {start_balance:.2f} USDC{RESET}")

        # Start the scouting thread
        scout_thread = threading.Thread(target=self.scout_task, daemon=True)
        scout_thread.start()

        # Start the position-checking thread
        check_thread = threading.Thread(target=self.position_check_loop, daemon=True)
        check_thread.start()

        if self.bot_type == "sniper":
            potential_entries_thread = threading.Thread(
                target=self.analyze_potential_entries_loop,
                daemon=True
            )
            potential_entries_thread.start()

        # Keep the main thread alive indefinitely
        while True:
            time.sleep(1)


def main():
    bot = BaseBot(
        wallet_address=Config.SNIPER_WALLET_ADDRESS,
        strategy_function="sniper",
        birdeye_api_key=Config.BIRDEYE_API_KEY,
        private_key_base58=Config.SNIPER_KEY_BASE58,
        rpc_url=Config.RPC_URL,
        starting_usdc_balance=1000.0,  # Start with 1,000 USDC
        max_open_trades=5,
        usd_per_trade=5.0,         # 5 minutes
        scout_interval=(50, "seconds"),
        position_check_interval=5,   # 5 seconds
        paper_trading=False,          # Enable paper trading
        trade_opened_less_than_x_mins_ago=180,  # 3 hours
        random_sample_size=20,
        sma_window_hours=24,
    )
    # res = bot.trade_logger.load_positions_from_db(bot.wallet_address, bot.bird_client.fetch_balances(bot.wallet_address), negligible_value_threshold=0.05)
    logging.info(f"Market is bearish: {bot.is_market_bearish()}")

    # usdc_amount = 1
    # token_address = "63LfDmNb3MQ8mw9MtZ2To9bEA2M71kZUUGq5tiJxcqj9"
    # position_id = bot.get_or_create_position_id(token_address)

    # tx_sig, status, exact_out_amount, block_time = swap_tokens(
    #             private_key_base58=bot.private_key_base58,
    #             rpc_url=bot.rpc_url,
    #             input_mint=USDC_MINT_ADDRESS,  # USDC
    #             input_mint_decimals=6,
    #             output_mint=token_address,
    #             amount_in=usdc_amount,
    #             slippage_bps=250,
    #             max_retries=3,
    #             retry_delay=10,
    #             bird_client=bot.bird_client,
    #             local_holdings=bot.holdings,
    #             holdings_lock=bot.holdings_lock,
    #             wallet_address=bot.wallet_address
    #         )

    # if status != "finalized" or not tx_sig:
    #     logging.error("Swap ultimately failed even after BirdEye fallback.")
    #     return 0.0

    # block_time = datetime.fromtimestamp(block_time, timezone.utc)

    # entry_price = usdc_amount / exact_out_amount
    # entry_time = timestamp = block_time
    # logging.info(f"{GREEN}Bought {exact_out_amount} (${usdc_amount}) of {token_address}: tx({tx_sig}){RESET}")

    # metadata = fetch_token_overview_and_metadata(token_address)

    # ticker = metadata.get("ticker", "")
    # token_name = metadata.get("name", "")


    # bot.trade_logger.record_trade(
    #     position_id=position_id,
    #     timestamp=block_time,
    #     token_address=token_address,
    #     ticker = ticker,
    #     token_name = token_name,
    #     tx_signature=tx_sig,
    #     entry_exit_price=entry_price,
    #     amount=exact_out_amount,
    #     buy_sell="buy",
    #     wallet_address = bot.wallet_address
    #     )

    # bot.trade_logger.upsert_position(
    #     position_id=position_id,
    #     entry_time= entry_time,
    #     entry_amount=exact_out_amount,
    #     entry_price=entry_price,
    #     partial_sold_cumulative=0.0,
    #     last_trade_time=entry_time,
    #     token_address=token_address,
    #     ticker_symbol=ticker,
    #     token_name=token_name,
    #     blockchain="solana",
    #     amount_holding=bot.holdings[token_address]["balance"],  # total holdings now
    #     amount_sold=None,
    #     realized_pnl=None,
    #     trade_status="open",
    #     type="market",
    #     wallet_address = bot.wallet_address,
    #     stoploss_price=None,
    #     max_recorded_price=None,
    # )


    # bot.run()
    # pos = bot.get_or_create_position_id("NuSvbWzz2QaqepZSyj8MhyvVv1E4W9AxnSge7G8pump")

if __name__ == "__main__":
    main()
