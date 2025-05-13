import time, sys, os, logging, threading, warnings, signal, pandas as pd
from typing import List, Tuple, Dict, Any
warnings.simplefilter(action='ignore', category=FutureWarning)
from datetime import datetime, timezone

from trader.agentic.cult_agent import CultAgent
from trader.agentic.sentiment_agent import *
from trader.agentic.src.subreddit_scout import SubredditScout
from trader.agentic.utils import *
from trader.database.supa_client import ( insert_row, update_row, execute_sql_query )
from trader.database.log_utils import DatabaseLogger
from trader.core.base_robot import *
from trader.sniper.analysis import analyze_token_drawdowns_6m

from solana.rpc.api import Client
from solders import keypair as Keypair
from trader.config import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class SentimentConfidencePairStruct(BaseModel):
    score: float
    confidence: float

class CultTrader:
    """
    A trading bot that scouts for new tokens, opens positions,
    and applies a “sentiment aware” partial selling strategy with trailing stops.

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
        scout_interval: Tuple[int, Literal['seconds', 'minutes', 'hours']] = (6, 'hours'),
        position_check_interval: int = 30,
        analyze_potential_entries_interval: int = 60,
        paper_trading: bool = False,
        table_name: str = "trades",
        bot_type: str = "sentitrader",
        mc_bounds: Tuple[float, float] = (50_000, 20_000_000),
        min_trades_in_30m: int = 30,
        max_drawdown_allowed: float = 70.0,
        top10_holder_percent_limit: float = 0.30,
        creation_max_days_ago: int = None,
        creation_min_days_ago: int = 90,   # token must be created within these many days
        creation_max_hours_ago: int = 2,   # token must NOT be created too recently, e.g. last 2 hours
        trade_opened_less_than_x_mins_ago: int = 60, # trade must be opened within these many minutes
        random_sample_size: int = 10,
        retention_probability: float = 0.95,
        buy_cooldown_seconds: int = 1800, # 30 minutes
        sell_cooldown_seconds: int = 604800, # 1 week
        next_tradable_hours: int = 48,
        sma_window_hours: int = 48,
    ):
        """
        Initialize the PiggyBackSniper with necessary configuration.
        """
        self.bot = BaseBot(
            strategy_function=self.apply_sentiment_aware_strategy,
            analyze_function = self.sentiment_aware_analyze_and_buy,
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
            buy_cooldown_seconds=buy_cooldown_seconds,
            sell_cooldown_seconds=sell_cooldown_seconds,
            next_tradable_hours=next_tradable_hours,
            sma_window_hours=sma_window_hours,
        )

        # sent, cultism etc
        self.sentinel = SentimentAgent()
        self.cultist = None
        self.db_logger = DatabaseLogger(table_name, bot_type)
        self.subreddit = None


    
    def run(self):
        self.bot.run()


    # --------------------------------------------------------------------
    # AI Methods
    # --------------------------------------------------------------------

    def globalise_subreddit(self, ticker, token_name):
        """
        Find subreddit associated with token and store as global variable
        """

        try:
            if not self.subreddit:
                self.subreddit = SubredditScout().find_subreddit_ai(ticker, token_name)
                print(f"Subreddit found: {self.subreddit}")
                logging.info(f"Subreddit found: {self.subreddit}")

        except Exception as e:
            logging.error(f"Subreddit scouting failed. Error: {e}")


    # maybe have it run on each source separately -- perform bias factor calc manually
    # flipside is we want single model to see biggest picture + single source of truth
    def capture_sentiment(
            self, 
            token_address: str,
            # average_holder_time: float = None,
            ):

        query = f"""
        SELECT sentiment_score, confidence_score, timestamp
        FROM sentiment
        WHERE token_address = '{token_address}'
        AND timestamp >= NOW() - INTERVAL '24 HOURS'
        ORDER BY timestamp DESC
        LIMIT 1
        """
        query = query.strip()
        sentiment_records = execute_sql_query(query)

        if sentiment_records and 'result' in sentiment_records[0]:
            latest = sentiment_records[0]['result']
            sentiment_score = latest.get('sentiment_score', None)
            confidence_score = latest.get('confidence_score', None)
            if sentiment_score is not None and confidence_score is not None:
                logging.info(f"Using cached sentiment score for {token_address}: {sentiment_score}")
                print(f"Using cached sentiment score for {token_address}: {sentiment_score}")
                
                return SentimentConfidencePairStruct(
                    score=sentiment_score, 
                    confidence=confidence_score
                    )
        
        if token_address in self.bot.token_metadata_cache:
            metadata = self.bot.token_metadata_cache[token_address]

        else:
            metadata = fetch_token_overview_and_metadata(token_address)
            self.bot.token_metadata_cache[token_address] = metadata

        ticker = metadata.get("ticker", "")
        token_name = metadata.get("name", "")
        num_holders = metadata.get("holders", "")
        volume_24h = metadata.get("volume_24h", None)
        volume_marketcap_ratio = metadata.get("volume_to_market_cap_ratio", None)
        buy_sell_ratio_24h = metadata.get("buy_sell_ratio_24h", None)
        holders_percent_increase_24h = metadata.get("holders_increase_ratio", None)
        avg_holders_dsitribution = metadata.get("holders_to_market_cap_ratio", None)
        marketcap = metadata.get("market_cap", None)
        links = metadata.get("urls", [])
        
        self.globalise_subreddit(ticker, token_name)

        payload = SentimentAgentInput(
                ticker=ticker,
                token_name = token_name,
                max_tweets=10,
                max_reddit_posts=2,
                hours=24,
                market_cap=marketcap,
                avg_holders_distribution=avg_holders_dsitribution,
                holders_percent_increase_24h=holders_percent_increase_24h,
                volume_marketcap_ratio=volume_marketcap_ratio,
                volume_24h=volume_24h,
                num_holders=num_holders,
                buy_sell_ratio_24h=buy_sell_ratio_24h,
                found_subreddit=self.subreddit,
                links = links
            )

        try:
            results = self.sentinel.analyze_ticker_sentiment(payload)
            self.db_logger.log_sentiment(
                ticker=ticker,
                token_name=token_name,
                token_address=token_address,
                market_cap=marketcap,
                avg_holders_distribution=avg_holders_dsitribution,
                result=results.sentiment,
                num_holders=num_holders,
                volume_24h=volume_24h,
                buy_sell_ratio_24h=buy_sell_ratio_24h,
                volume_marketcap_ratio=volume_marketcap_ratio,
                holders_percent_increase_24h=holders_percent_increase_24h,
                links=links
            )

            return results.sentiment

        except Exception as e:
            logging.error(f"Failed to receive sentiment of {ticker}. Error: {e}")
            return None

    def capture_cultism(self, ticker, token_name, token_address: str, num_holders):

        query = f"""
        SELECT cult_score, timestamp
        FROM cultism
        WHERE token_address = '{token_address}'
        AND timestamp >= NOW() - INTERVAL '7 Days'
        ORDER BY timestamp DESC
        LIMIT 1
        """
        query = query.strip()
        cult_records = execute_sql_query(query)

        if cult_records and 'result' in cult_records[0]:
            latest_cultism = cult_records[0]['result']
            cult_score = latest_cultism.get('overall_score', None)

            if cult_score is not None:
                logging.info(f"Using cached cultism score for {token_address}: {cult_score}")
                return cult_score

        try:
            verdict = self.cultist.search_cultism(ticker, token_name, num_holders)
            self.db_logger.log_cultiness(ticker, token_address, token_name, verdict)
            return verdict.score

        except Exception as e:
            logging.error(f"Failed to retrieve cultism of {ticker}. Error: {e}")

            return 0.0
    

    # ---------------------------------------------------------
    # SENTIMENT-AWARE BUY
    # ---------------------------------------------------------

    def sentiment_aware_analyze_and_buy(self, candidate_tokens: list[TokenMetadata]):
        """
        Analyze candidate tokens and buy them in parallel.
        """
        if not candidate_tokens:
            return

        tokens_with_analysis = self.ai_analyze_tokens(self.bot.bird_client, candidate_tokens, self.bot.trade_opened_less_than_x_mins_ago)

        return tokens_with_analysis
    
    
    def ai_analyze_tokens(self, client: BirdEyeClient, tokens: List[TokenMetadata], trade_opened_less_than_x_mins_ago: int=60):

        for token in tokens:
            address = token.address
            if address in self.bot.token_metadata_cache:
                metadata = self.bot.token_metadata_cache[address]

            else:
                metadata = fetch_token_overview_and_metadata(address)
                self.bot.token_metadata_cache[address] = metadata
            
            # TOKEN CATEGORISATION
            # categorisation -- make sure filtered to solana first (should be done in scout method)
            print(f"Categorising token {metadata.get('ticker')}")
            logging.info(f"Categorising token {metadata.get('ticker')}")
            result = categorise_token(metadata.get("ticker"), metadata.get("name"), metadata.get("logoURI"), metadata.get("description"))

            if result.category not in ("Cult", "Utility/Layer1/Layer2"):
                print(f"Skipping token {metadata.get('ticker')} as it is not a cult or utility token")
                logging.info(f"Skipping token {metadata.get('ticker')} as it is not a cult or utility token")
                continue
            logging.info(f"Categorised token {metadata.get('ticker')} as {result.category}")
            print(f"Categorised token {metadata.get('ticker')} as {result.category}")

            # DRAWDOWN-RECOVERY EVENTS
            # drawdowns -- whhere does new dd event begin
            print(f"Analyzing drawdowns for token {metadata.get('ticker')}")
            logging.info(f"Analyzing drawdowns for token {metadata.get('ticker')}")
            dd_events = analyze_token_drawdowns_6m(
                                                client=client,
                                                address=address,
                                                # drop_threshold=0.55,
                                                recovery_threshold=0.95,
                                                candles="1D"
                                                )
            
            if not isinstance(dd_events, dict):
                raise TypeError(f"Expected dd_events to be a dict, got {type(dd_events)}")
            
            drawdown_count = dd_events.get("num_drawdowns", 0)
            drawdown_resilience_score = dd_events.get("drawdown_resilience_score", 0.0)
            logging.info(f"Drawdown count: {drawdown_count}, Drawdown resilience score: {drawdown_resilience_score}")
            print(f"Drawdown count: {drawdown_count}, Drawdown resilience score: {drawdown_resilience_score}")

            if drawdown_count < 2:
                continue

            # SENTIMENT ANALYTICS       
            sentiment = self.capture_sentiment(address) 
            logging.info(f"Sentiment score for {address}: {sentiment.score:.2f}")
            print(f"Sentiment score for {address}: {sentiment.score:.2f}")
                    
            # === Score-based buy decision === removed cultiness (FOR NOW)
            s_score, dd_score, avg_score = self.compute_buy_scores(
                sentiment=sentiment, 
                drawdown_resilience_score=drawdown_resilience_score,
                dd_score_weight=0.5,
                sentiment_weight=0.5
                )

            print(f"Calculated scores: Sentiment score: {s_score:.2f}, Drawdown score: {dd_score:.2f}")
            logging.info(f"Calculated scores: sentiment score: {s_score:.2f}, Drawdown score: {dd_score:.2f}")
            print(f"Calculated average score: {avg_score:.2f}")
            logging.info(f"Calculated average score: {avg_score:.2f}")
            if not self.decide_buy(s_score, dd_score, avg_score, avg_threshold=0.6, min_threshold=0.4):
                logging.info(
                    f"Skipping {address} => (sentiment={s_score:.2f}, drawdown={dd_score:.2f}) did not meet thresholds."
                )
                print(f"Skipping {address} => (sentiment={s_score:.2f}, drawdown={dd_score:.2f}) did not meet thresholds.")
                continue

            # how many usdc to allocate -- default is usd to trade. max is 2x
            usd_to_buy = self.compute_buy_size(avg_score, self.bot.usd_per_trade)
            print(f"Computed buy size: {usd_to_buy:.2f} USDC")
            logging.info(f"Computed buy size: {usd_to_buy:.2f} USDC")
            if usd_to_buy <= 0:
                logging.info(f"Computed buy size is 0. Skipping token {address}.")
                print(f"Computed buy size is 0. Skipping token {address}.")
                continue
            
            if token.tx_time and isinstance(token.tx_time.timestamp(), datetime):
                time_since_tx = time.time() - token.tx_time.timestamp()
                if time_since_tx < trade_opened_less_than_x_mins_ago * 60:
                    token.open_new_position = True
        
        return tokens
    

    def decide_buy(
        self,
        sentiment_score: float,
        drawdown_score: float,
        avg_score: float,
        avg_threshold: float = 0.7,
        min_threshold: float = 0.4
    ) -> bool:
        
        if drawdown_score < min_threshold or sentiment_score < min_threshold:
            return False

        scores = [sentiment_score, drawdown_score]

        if avg_score >= avg_threshold and all(s >= min_threshold for s in scores):
            return True
        return False
    

    # compute buy size based on avg score
    def compute_buy_size(
        self,
        avg_score: float,
        base_usd_per_trade: float
    ) -> float:
        # e.g.  if avg_score=0.7 => buy exactly base
        #       if avg_score=1.0 => buy 2x base
        #       linear scale in between
        multiplier = 1.0 + (avg_score - 0.7) * 2.5
        multiplier = max(0.0, min(multiplier, 2.0)) # between 0 and 2x buys

        potential_buy = base_usd_per_trade * multiplier
        logging.info(f"Potential buy: {potential_buy:.2f} USDC based on avg_score={avg_score:.2f}")
        print(f"Potential buy: {potential_buy:.2f} USDC based on avg_score={avg_score:.2f}")

        total_holdings_value = self.bot.compute_total_holdings_value()  # in USDC
        print(f"Total holdings value: {total_holdings_value:.2f} USDC")
        logging.info(f"Total holdings value: {total_holdings_value:.2f} USDC")
        if total_holdings_value <= 0:
            return 0.0

        max_allocation = 0.10 * total_holdings_value

        if potential_buy > max_allocation:
            potential_buy = max_allocation

        # req higher average score for buys above 5% holdings
        if potential_buy > 0.05 * total_holdings_value and avg_score < 0.8:
            potential_buy = 0.05 * total_holdings_value
        
        if potential_buy < base_usd_per_trade:
            potential_buy = base_usd_per_trade

        return potential_buy


    def compute_buy_scores(
        self,
        sentiment: SentimentResult,
        drawdown_resilience_score: float,
        dd_score_weight: float = 0.5,  # 50% weight for drawdown resilience
        sentiment_weight: float = 0.5  # 50% weight for sentiment
    ) -> Tuple[float, float, float]:
        """
        Compute buy scores based on:
        1. `sentiment_score`: Indicates market sentiment.
        2. `drawdown_resilience_score`: Indicates token’s ability to recover from drops.
        3. Number of drawdowns (optional filter for insufficient historical events).

        Parameters:
        - `drawdown_resilience_score`: Between 0.0 (poor resilience) and 1.0 (excellent resilience).
        - `num_drawdowns`: Total number of recorded drawdowns.
        - `min_dd_count_for_good_score`: Minimum drawdowns required for a token to be eligible for consideration.
        - `min_resilience_for_good_score`: Minimum resilience score to consider a token.
        - `dd_score_weight` and `sentiment_weight`: Weights for combining sentiment and resilience.

        Returns:
        - `sentiment_score`, `drawdown_resilience_score`, `average_score`.
        """

        sentiment_score = sentiment.score

        avg_score = (
            dd_score_weight * drawdown_resilience_score
            + sentiment_weight * sentiment_score
        )

        logging.info(
            f"Sentiment Score: {sentiment_score:.2f}, "
            f"Drawdown Resilience Score: {drawdown_resilience_score:.2f}, "
            f"Average Score: {avg_score:.2f}"
        )

        return sentiment_score, drawdown_resilience_score, avg_score


    # ---------------------------------------------------------
    # SENTIMENT-AWARE SELL
    # ---------------------------------------------------------

    def decide_sentiment_aware_sell(
        self,
        results: SentimentResult,
        alpha: float = 0.7  # 70% sent -- 30% confidence weighting -- set to 1 to remove confidence score weighting
    ) -> Tuple[float, str]:
        """
        Decide on a fraction to sell (0.0 to 1.0) based on weighted scaled sentiment.
        Returns (fraction_to_sell, reason).
        - If fraction_to_sell == 1.0 => full exit
        - If fraction_to_sell == 0.0 => do nothing
        """
        sentiment = results.score
        confidence = results.confidence

        scaled_sentiment = alpha * sentiment + (1 - alpha) * (sentiment * confidence)

        if scaled_sentiment < 0.15:
            return (1.0, f"Ultra negative scaled sentiment < 0.15 => Full exit (scaled={scaled_sentiment:.2f}, raw={sentiment:.2f}, confidence={confidence:.2f})")

        # if scaled_sentiment < 0.35:
        #     return (0.25, f"Scaled sentiment < 0.35 => partial exit (25%) (scaled={scaled_sentiment:.2f}, raw={sentiment:.2f}, confidence={confidence:.2f})")

        # if scaled_sentiment < 0.50:
        #     return (0.10, f"Scaled sentiment < 0.50 => partial exit (10%) (scaled={scaled_sentiment:.2f}, raw={sentiment:.2f}, confidence={confidence:.2f})")

        # Otherwise => no sell
        return (0.0, f"No sell: scaled sentiment={scaled_sentiment:.2f} (raw={sentiment:.2f}, confidence={confidence:.2f})")



    def apply_sentiment_aware_strategy(self, idx: int, current_price: float):
        with self.bot.positions_lock:
            if idx not in self.bot.positions_df.index:
                return  # Already removed or closed
            row = self.bot.positions_df.loc[idx]

        if row["status"] != "open":
            return

        token_address = row["token_address"]
        entry_price   = row["entry_price"]
        original_amt  = row["entry_amount"]
        partial_sold  = row["partial_sold_cumulative"]
        realized_pnl  = row["realized_profit_usd"]
        max_price     = row["max_price"]

        # Update max price if needed
        if current_price > max_price:
            max_price = current_price

        # ---------------------------------------
        # 1) Price-Based Partial Sell
        # ---------------------------------------
        # partial_sells_table = [
        #     (1.5, 0.10),  # 1.5x => sell 10% total
        #     (2.0, 0.25),  # 2.0x => sell 25% total
        #     (3.0, 0.50),  # 3.0x => sell 50% total
        #     (4.0, 0.75),  # 4.0x => sell 75% total
        #     (5.0, 1.00),  # 5.0x => fully exit
        # ]

        partial_sells_table = [
            (1.5, 0.15), 
            (2.5, 0.25), 
            (3.0, 0.30),  
            (4.0, 0.35),  
            (5.0, 0.40),  
            (6.0, 0.45),  
            (7.0, 0.50),
            (8.0, 0.55),
            (9.0, 0.60),
            (10.0, 0.65),
            (15.0, 0.90),
            (20.0, 1.0),   
        ]

        current_factor = (current_price / entry_price) if entry_price > 0 else 0
        max_factor     = (max_price / entry_price) if entry_price > 0 else 0

        total_fraction_to_sell = 0.0
        for (threshold_factor, fraction_sold_target) in partial_sells_table:
            if max_factor >= threshold_factor and partial_sold < fraction_sold_target:
                # Sell enough to reach fraction_sold_target
                diff = fraction_sold_target - partial_sold
                if diff > 0:
                    total_fraction_to_sell += diff
                    partial_sold = fraction_sold_target
        
        logging.info(f"Applying sentiment-aware strategy for token {row['token_address']}")


        if total_fraction_to_sell > 0:
            tokens_to_sell = original_amt * total_fraction_to_sell
            part_profit = self.bot.sell_token(
                token_address=token_address,
                token_amount=tokens_to_sell,
                partial_sold_cumulative=float(partial_sold),
                realized_pnl=float(realized_pnl),
                stoploss_price=None,
                max_recorded_price=float(max_price)
            )
            if part_profit:
                realized_pnl += part_profit
                logging.info(f"Price-based partial exit triggered. Sold fraction: {total_fraction_to_sell*100:.1f}%")

            # If partial_sold >= 1.0 => fully out. Remove from DF and return.
            if partial_sold >= 1.0:
                net_profit = realized_pnl - (entry_price * original_amt)
                self.bot.log_exit_and_remove(idx, net_profit, realized_pnl, partial_sold, max_price,
                                             reason="Fully sold from partial sells")
                return

        # ---------------------------------------
        # 2) Trailing Stop
        # ---------------------------------------
        # Example trailing stop factor: 70% of max if sentiment is unknown
        trailing_stop_factor = 0.70
        stop_price = max_price * trailing_stop_factor

        if current_price < stop_price:
            leftover_frac = 1.0 - partial_sold
            if leftover_frac > 0:
                tokens_leftover = original_amt * leftover_frac
                part_profit = self.bot.sell_token(
                    token_address=token_address,
                    token_amount=tokens_leftover,
                    partial_sold_cumulative=1.0,
                    realized_pnl=realized_pnl,
                    stoploss_price=stop_price,
                    max_recorded_price=max_price
                )
                if part_profit:
                    realized_pnl += part_profit
                net_profit = realized_pnl - (entry_price * original_amt)
                self.bot.log_exit_and_remove(idx, net_profit, realized_pnl, partial_sold, max_price,
                                            reason=f"Trailing stop triggered @ {stop_price:.4f}")
            return

        # ---------------------------------------
        # 3) Sentiment Override
        # ---------------------------------------
        sentiment = self.capture_sentiment(token_address)
        fraction_to_sell, reason = self.decide_sentiment_aware_sell(sentiment)
        logging.info(f"Sentiment-based sell: {fraction_to_sell*100:.1f}% for reason:\n {reason}")

        if fraction_to_sell >= 1.0:
            # Full exit
            leftover_frac = 1.0 - partial_sold
            if leftover_frac > 0:
                tokens_leftover = original_amt * leftover_frac
                part_profit = self.bot.sell_token(
                    token_address=token_address,
                    token_amount=tokens_leftover,
                    partial_sold_cumulative=float(1.0),
                    realized_pnl=float(realized_pnl),
                    stoploss_price=None,
                    max_recorded_price=float(max_price)
                )
                if part_profit:
                    realized_pnl += part_profit
            net_profit = realized_pnl - (entry_price * original_amt)
            self.bot.log_exit_and_remove(idx, net_profit, realized_pnl, partial_sold, max_price, reason=reason)
            return

        elif fraction_to_sell > 0.0:
            # Partial exit
            leftover_frac  = 1.0 - partial_sold
            partial_to_sell = fraction_to_sell * leftover_frac
            if partial_to_sell > 0:
                tokens_to_sell = original_amt * partial_to_sell
                part_profit = self.bot.sell_token(
                    token_address=token_address,
                    token_amount=tokens_to_sell,
                    partial_sold_cumulative=float(partial_sold + fraction_to_sell),
                    realized_pnl=float(realized_pnl),
                    stoploss_price=None,
                    max_recorded_price=float(max_price)
                )
                if part_profit:
                    realized_pnl += part_profit
                    partial_sold += fraction_to_sell
                    logging.info(f"Sentiment-based partial exit => {reason}")

        # ---------------------------------------
        # 4) Update DataFrame
        # ---------------------------------------
        with self.bot.positions_lock:
            if idx in self.bot.positions_df.index:
                self.bot.positions_df.loc[idx, "partial_sold_cumulative"] = partial_sold
                self.bot.positions_df.loc[idx, "realized_profit_usd"]     = realized_pnl
                self.bot.positions_df.loc[idx, "max_price"]               = max_price
                self.bot.positions_df.loc[idx, "last_price"]              = current_price


    def sentiment_aware_trailing_stop(self, sentiment, base_stop_factor = 0.50):

        if sentiment > 0.70:
            trailing_stop_factor = 0.70 # higher stop for v good sentiment
        elif sentiment < 0.50:
            trailing_stop_factor = 0.40 # strict stop for lower sentiment
        else:
            trailing_stop_factor = base_stop_factor
        
        return trailing_stop_factor


def main():
    bot = CultTrader(
        birdeye_api_key=Config.BIRDEYE_API_KEY,
        wallet_address=Config.sentitrader_WALLET_ADDRESS,
        private_key_base58=Config.sentitrader_KEY_BASE58,
        mc_bounds=(5_000_000, 20_000_000),
        rpc_url=Config.RPC_URL,
        starting_usdc_balance=1000.0,  # Start with 1,000 USDC
        max_open_trades=10,
        usd_per_trade=5.0,
        scout_interval=(6, "hours"),  # 6 hours
        # scout_interval=(10, "seconds"),
        position_check_interval=30,   # whilst running -- 300 second position checks
        paper_trading=False,
        # trade_opened_less_than_x_mins_ago=2880, # 48 hrs # None => all trades -- hits rate limits so will do like 300
        trade_opened_less_than_x_mins_ago=300,
        random_sample_size=30,
        next_tradable_hours=48,
        sma_window_hours=48,
    )
    bot.run()
    # bot.bot.scout_and_buy()
    # res = bot.sentiment_aware_analyze_and_buy([TokenMetadata(address="63LfDmNb3MQ8mw9MtZ2To9bEA2M71kZUUGq5tiJxcqj9")])

if __name__ == "__main__":
    main()
