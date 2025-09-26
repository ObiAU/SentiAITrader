import logging
import math
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, field_validator

from database.supa_client import insert_row, execute_sql_query, upsert_row_static
from trader.agentic.data_models import CultinessResult, SentimentResult


class TokenInfo(BaseModel):
    address: str
    name: Optional[str] = None
    symbol: Optional[str] = None
    decimals: int
    balance: str
    uiAmount: float
    chainId: str
    logoURI: Optional[str] = None  # nonetype to handle logoURI
    priceUsd: Optional[float] = None
    valueUsd: Optional[float] = None

    @field_validator("balance", mode="before")
    def ensure_string_balance(cls, value):
        return str(value) if isinstance(value, (int, float)) else value


class DatabaseLogger:
    def __init__(self, table_name: str, bot_executed: Optional[str] = None):
        self.TABLE_NAME = table_name
        self.BOT_EXECUTED = bot_executed

        self.positions_df = pd.DataFrame()

    def get_max_recorded_price(
        self, token_address: str, position_id: int
    ) -> Optional[float]:
        query = f"""
            SELECT max_recorded_price FROM positions WHERE token_address = '{token_address}' and position_id = {position_id} LIMIT 1
        """
        query = query.strip()
        try:
            result = execute_sql_query(query)
            if result:
                resultant = result[0].get("result")
            return resultant.get("max_recorded_price", 0.0)
        except Exception as e:
            logging.error(
                f"Failed to get max recorded price for {token_address} and position_id {position_id}. Error: {e}"
            )
            return 0.0

    def upsert_position(
        self,
        position_id: int,
        wallet_address: str,
        token_address: str,
        entry_time: Optional[datetime] = None,
        entry_price: Optional[float] = None,
        entry_amount: Optional[float] = None,
        next_tradable: Optional[datetime] = None,
        partial_sold_cumulative: Optional[float] = None,
        last_trade_time: Optional[datetime] = None,
        ticker_symbol: Optional[str] = None,
        token_name: Optional[str] = None,
        blockchain: Optional[str] = "solana",
        amount_holding: Optional[float] = None,
        amount_sold: Optional[float] = None,
        stoploss_price: Optional[float] = None,
        max_recorded_price: Optional[float] = None,
        realized_pnl: Optional[float] = None,
        trade_status: Optional[Literal["open", "closed"]] = None,
        type: Optional[str] = "market",
    ):
        """
        Upserts a position by its primary key, position_id.
        Updates the row if it exists, otherwise logs an error.

        Args:
            position_id (int): Primary key of the position to update.
            All other parameters are optional and will be updated if provided.
        """
        # type safety
        position_id = int(position_id)

        if entry_price:
            entry_price = float(entry_price) if not math.isnan(entry_price) else 0.0

        if entry_amount:
            entry_amount = float(entry_amount) if not math.isnan(entry_amount) else 0.0

        if partial_sold_cumulative:
            partial_sold_cumulative = (
                float(partial_sold_cumulative)
                if not math.isnan(partial_sold_cumulative)
                else 0.0
            )

        if max_recorded_price:
            max_recorded_price = (
                float(max_recorded_price) if not math.isnan(max_recorded_price) else 0.0
            )

        if amount_holding:
            amount_holding = (
                float(amount_holding) if not math.isnan(amount_holding) else 0.0
            )

        if amount_sold:
            amount_sold = float(amount_sold) if not math.isnan(amount_sold) else 0.0

        if realized_pnl:
            realized_pnl = float(realized_pnl) if not math.isnan(realized_pnl) else 0.0

        if max_recorded_price:
            db_recorded_price = self.get_max_recorded_price(token_address, position_id)
            if db_recorded_price and db_recorded_price > max_recorded_price:
                logging.info(
                    f"DB recorded price {db_recorded_price} is greater than max_recorded_price {max_recorded_price}. Setting max_recorded_price to None."
                )
                max_recorded_price = None

        trace = {
            "entry_time": (
                entry_time.isoformat()
                if isinstance(entry_time, datetime)
                else entry_time
            ),
            "last_trade_time": (
                last_trade_time.isoformat()
                if isinstance(last_trade_time, datetime)
                else last_trade_time
            ),
            "next_tradable": (
                next_tradable.isoformat()
                if isinstance(next_tradable, datetime)
                else next_tradable
            ),
            "token_address": token_address,
            "status": trade_status,
            "ticker_symbol": ticker_symbol,
            "entry_price": entry_price,
            "entry_amount": entry_amount,
            "partial_sold_cumulative": partial_sold_cumulative,
            "token_name": token_name,
            "blockchain": blockchain,
            "amount_holding": amount_holding,
            "amount_sold": amount_sold,
            "stoploss_price": stoploss_price,
            "max_recorded_price": max_recorded_price,
            "realized_pnl": realized_pnl,
            "type": type,
            "bot_executed": self.BOT_EXECUTED,
            "wallet_address": wallet_address,
        }

        # update_fields = {k: (int(v) if isinstance(v, np.int64) else v) for k, v in trace.items() if v is not None}
        update_fields = {
            k: (
                float(v)
                if isinstance(v, (np.int64, np.int32))
                else (int(v) if isinstance(v, np.integer) else v)
            )
            for k, v in trace.items()
            if v is not None
        }

        if not update_fields:
            logging.warning("No fields provided to update.")
            return

        try:
            update_fields["position_id"] = int(position_id)

            upsert_row_static("positions", update_fields)
            logging.info(
                f"Successfully upserted position with ID {position_id}: {update_fields}"
            )
        except Exception as e:
            logging.error(
                f"Failed to upsert position with ID {position_id}. Error: {e}"
            )
            print(f"Failed to upsert position with ID {position_id}. Error: {e}")

    def record_trade(
        self,
        position_id: int,
        timestamp: datetime,
        token_address: str,
        wallet_address: str,
        ticker: str = None,
        token_name: str = None,
        tx_signature: Optional[str] = None,
        entry_exit_price: Optional[float] = None,
        amount: Optional[float] = None,
        buy_sell: Optional[str] = None,
        type: str = "market",
        blockchain: str = "solana",
    ):
        timestamp = timestamp.isoformat()

        if entry_exit_price:
            entry_exit_price = (
                float(entry_exit_price) if not math.isnan(entry_exit_price) else 0.0
            )

        if amount:
            amount = float(amount) if not math.isnan(amount) else 0.0

        trace = {
            "position_id": position_id,
            "timestamp": timestamp,
            "token_address": token_address,
            "ticker_symbol": ticker,
            "token_name": token_name,
            "blockchain": blockchain,
            "transaction_signature": tx_signature,
            "entry_exit_price": entry_exit_price,
            "amount": amount,
            "buy_sell": buy_sell,
            "type": type,
            "bot_executed": self.BOT_EXECUTED,
            "wallet_address": wallet_address,
        }

        insert_fields = {
            k: (
                float(v)
                if isinstance(v, (np.int64, np.int32))
                else (int(v) if isinstance(v, np.integer) else v)
            )
            for k, v in trace.items()
            if v is not None
        }

        insert_fields["position_id"] = int(position_id)

        try:
            insert_row("trades", insert_fields)
            logging.info(
                f"Successfully recorded {buy_sell} trade for ${amount} of {token_address} at {timestamp}."
            )

        except Exception as e:
            logging.error(
                f"Failed to record {buy_sell} trade of {token_address}. Error: {e}"
            )

    def log_cultiness(
        self, ticker: str, token_address: str, token_name: str, result: CultinessResult
    ):
        current_time = datetime.now().isoformat()

        last_7_days_scores = self._fetch_recent_scores(ticker, days=7)

        if last_7_days_scores:
            overall_score = sum(last_7_days_scores) / len(last_7_days_scores)
        else:
            overall_score = result.score

        trace = {
            "timestamp": current_time,
            "ticker_symbol": ticker,
            "token_name": token_name,
            "token_address": token_address,
            "cult_score": result.score,
            "overall_score": overall_score,
            "analysis": result.analysis,
            "warnings": result.warnings,
        }

        try:
            insert_row(self.TABLE_NAME, trace)
            logging.info(
                f"Successfully logged cultiness for {ticker} at {current_time}."
            )

        except Exception as e:
            logging.error(f"Failed to log cultiness. Error: {e}")

    def log_sentiment(
        self,
        ticker: str,
        token_address: str,
        token_name: str,
        result: SentimentResult,
        num_holders: int = None,
        volume_24h: str = None,
        buy_sell_ratio_24h: float = None,
        holders_percent_increase_24h: float = None,
        volume_marketcap_ratio: float = None,
        avg_holders_distribution: float = None,
        market_cap: float = None,
        links: List[str] = None,
    ):
        timestamp = datetime.now().isoformat()
        trace = {
            "timestamp": timestamp,
            "ticker_symbol": ticker,
            "token_name": token_name,
            "token_address": token_address,
            "sentiment_score": result.score,
            "confidence_score": result.confidence,
            "overall": result.overall,
            "warnings": result.warnings,
            "holders_percentage_increase_24h": holders_percent_increase_24h,
            "volume_to_marketcap_ratio": volume_marketcap_ratio,
            "avg_holders_distribution": avg_holders_distribution,
            "market_cap": market_cap,
            "volume_24h": volume_24h,
            "num_holders": num_holders,
            "buy_sell_ratio_24h": buy_sell_ratio_24h,
            "urls": links,
        }
        try:
            insert_row("sentiment", trace)
            logging.info(f"Successfully logged sentiment for {ticker} at {timestamp}.")

        except Exception as e:
            logging.error(f"Failed to log sentiment. Error: {e}")

    def _fetch_recent_scores(self, ticker: str, days: int) -> List[float]:
        cutoff_time = datetime.now() - timedelta(days=days)

        query = f"""
            SELECT cult_score
            FROM {self.TABLE_NAME}
            WHERE ticker_symbol = %s AND timestamp >= %s
        """
        query = query.strip()
        try:
            results = execute_sql_query(query, (ticker, cutoff_time))
            return [row["cult_score"] for row in results]

        except Exception as e:
            logging.error(f"Failed to fetch recent scores for {ticker}. Error: {e}")
            return []

    # --------------------------------------------------------------------
    # Initialization Methods
    # --------------------------------------------------------------------

    def find_position_id_small(self, token_address: str) -> int:
        """
        Returns the existing position_id for a token, or creates one if none exists.
        """
        token_address = token_address.strip()
        query = f"""
            SELECT position_id FROM positions WHERE token_address = '{token_address}' and bot_executed = '{self.BOT_EXECUTED}' LIMIT 1
        """
        query = query.strip()
        try:
            result = execute_sql_query(query)
        except Exception as e:
            logging.error(f"Failed to find position ID for {token_address}. Error: {e}")
            result = None

        if result:
            resultant = result[0].get("result")
            if resultant:
                position_id = resultant.get("position_id", None)
                logging.info(f"Position ID found in DB: {position_id}")
                return position_id

        try:
            insert_result = insert_row(
                "positions", {"token_address": token_address, "status": "open"}
            )
        except Exception as e:
            logging.error(f"Failed to insert row for token {token_address}. Error: {e}")
            insert_result = None

        if not insert_result:
            raise Exception(f"Failed to insert row for token {token_address}")

        new_position_id = insert_result[0]["position_id"]

        return new_position_id

    def qol_upsert_positions(self):
        query = f"""
                SELECT
                    position_id,
                    token_address,
                    amount_holding
                FROM positions
                WHERE status = 'closed'
                AND bot_executed = '{self.BOT_EXECUTED}'
            """
        query = query.strip()
        try:
            rows = execute_sql_query(query)
        except Exception as e:
            logging.error(f"Failed to execute query. Error: {e}")
            rows = None
        if not rows:
            logging.info("No closed positions found in DB.")
            return
        logging.info(f"Found {len(rows)} closed positions in DB.")
        for row in rows:
            try:
                result = row["result"]
                position_id = result["position_id"]
                token_address = result["token_address"]
                amount_holding = float(result.get("amount_holding", 0.0))
                if amount_holding != 0.0:
                    amount_holding = float(0.0)
                    upsert_row_static(
                        "positions",
                        {
                            "position_id": position_id,
                            "token_address": token_address,
                            "amount_holding": amount_holding,
                        },
                    )
            except Exception as e:
                logging.error(f"Failed to upsert position {position_id}. Error: {e}")
                continue

    def load_positions_from_db(
        self,
        wallet_address: str,
        balances: List[TokenInfo],
        barred_addresses: List[str] = [],
        negligible_value_threshold=0.05,
    ):
        # add USDC Mint and dummy SOL address to barred_addresses
        barred_addresses.extend(["EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"])

        try:
            self.qol_upsert_positions()
        except Exception as e:
            logging.error(f"Failed to upsert closed positions. Error: {e}")
            return

        try:
            query = f"""
                SELECT
                    position_id,
                    token_address,
                    ticker_symbol AS ticker,
                    token_name,
                    entry_price,
                    entry_amount,
                    amount_holding as current_amount,
                    partial_sold_cumulative,
                    realized_pnl AS realized_profit_usd,
                    max_recorded_price AS max_price,
                    status,
                    cast(entry_time as timestamp) as entry_time,
                    cast(next_tradable as timestamp) as next_tradable
                FROM positions
                WHERE status = 'open'
                AND bot_executed = '{self.BOT_EXECUTED}'
                AND wallet_address = '{wallet_address}'
            """
            query = query.strip()
            rows = execute_sql_query(query)

            onchain_bal_map = {}
            for b in balances:
                if b.address in barred_addresses:
                    continue

                onchain_amount = float(b.balance) / (10**b.decimals)
                token_value = onchain_amount * (b.priceUsd or 0.0)

                onchain_bal_map[b.address] = {
                    "amount": onchain_amount,
                    "symbol": b.symbol or "",
                    "name": b.name or "",
                    "decimals": b.decimals,
                    "priceUsd": b.priceUsd,
                    "balance": b.balance,
                    "value": token_value,
                }

            if not rows:
                logging.info(
                    f"No open positions in DB for wallet={wallet_address}. "
                    f"Constructing DataFrame from BirdEye balances..."
                )

                new_entries = []
                for b in balances:
                    if b.address in barred_addresses:
                        continue

                    chain_info = onchain_bal_map.get(b.address, {})
                    chain_amt = chain_info.get("amount", 0.0)
                    if chain_amt <= 0:
                        continue  # skip zero-balance tokens

                    position_id = self.find_position_id_small(b.address)

                    new_entries.append(
                        {
                            "position_id": position_id,
                            "token_address": b.address,
                            "ticker": b.symbol or "",
                            "token_name": b.name or "",
                            "entry_price": 0.0,
                            "entry_amount": chain_amt,
                            "current_amount": chain_amt,
                            "partial_sold_cumulative": 0.0,
                            "realized_profit_usd": 0.0,
                            "max_price": 0.0,
                            "status": "open",
                            "entry_time": datetime.now(timezone.utc),
                            "wallet_address": wallet_address,
                        }
                    )

                try:
                    upsert_row_static(
                        "positions",
                        {
                            "position_id": position_id,
                            "token_address": b.address,
                            "ticker_symbol": b.symbol,
                            "token_name": b.name,
                            "amount_holding": chain_amt,
                            "blockchain": "solana",
                            "status": "open",
                            "entry_time": datetime.now().isoformat(),
                            "wallet_address": wallet_address,
                            "type": "market",
                            "bot_executed": self.BOT_EXECUTED,
                        },
                    )
                except Exception as e:
                    logging.error(f"Failed to upsert new position. Error: {e}")

                df = pd.DataFrame(new_entries)
                logging.info(f"Built {len(df)} rows from on-chain balances.")
                return df

            extracted_rows = [r["result"] for r in rows]
            df = pd.DataFrame(extracted_rows)

            df["entry_time"] = pd.to_datetime(
                df["entry_time"], format="mixed", errors="coerce"
            )

            df["next_tradable"] = pd.to_datetime(
                df["next_tradable"], format="mixed", errors="coerce"
            )

            for i, row in df.iterrows():
                t_addr = row["token_address"]

                if t_addr not in onchain_bal_map:
                    if row["status"] != "closed":
                        logging.info(
                            f"Closing position_id={row['position_id']} (no on-chain balance)."
                        )
                        df.at[i, "status"] = "closed"

                    upsert_row_static(
                        "positions",
                        {
                            "position_id": row["position_id"],
                            "token_address": t_addr,
                            "amount_holding": 0.0,
                            "status": "closed",
                            "bot_executed": self.BOT_EXECUTED,
                        },
                    )
                    continue

                chain_amt = onchain_bal_map[t_addr]["amount"]
                chain_val = onchain_bal_map[t_addr]["value"]
                df.at[i, "current_amount"] = chain_amt
                df.at[i, "status"] = "open"

                if chain_val < negligible_value_threshold:
                    logging.info(
                        f"Closing position_id={row['position_id']} "
                        f"since value={chain_val:.2f} < {negligible_value_threshold}"
                    )
                    df.at[i, "status"] = "closed"
                    continue

                update_data = {
                    "position_id": row["position_id"],
                    "token_address": t_addr,
                    "amount_holding": chain_amt,
                    "status": df.at[i, "status"],
                    "bot_executed": self.BOT_EXECUTED,
                }
                upsert_row_static("positions", update_data)

            return df

        except Exception as e:
            logging.error(f"Error loading positions from DB: {e}")
            return pd.DataFrame()

    def load_token_metadata_cache(self):
        try:
            query = """
                SELECT token_address, 
                    ticker, 
                    token_name, 
                    decimals, 
                    holders -- if you store that in DB
                FROM token_metadata;
            """
            query = query.strip()
            rows = execute_sql_query(query)
            if not rows:
                logging.info("No token metadata found in DB.")
                return

            for row in rows:
                address = row["token_address"]
                self.token_metadata_cache[address] = {
                    "ticker": row.get("ticker"),
                    "name": row.get("token_name"),
                    "decimals": row.get("decimals", 0),
                    "holders": row.get("holders", 0),
                }

            logging.info(f"Loaded metadata for {len(rows)} tokens from DB.")

        except Exception as e:
            logging.error(f"Error loading token metadata from DB: {e}")


if __name__ == "__main__":
    from trader.config import Config

    SOL_ADDRESS = "SOL0x1111111111111111111111111111111111111111"
    WALLET_ADDRESS = Config.SNIPER_WALLET_ADDRESS
    TX_SIG = "5fpM4Ae52MDZA219mYuwLHMHq6UE9agzGaYVS9LJMhwWnfAZYGNWGBfgFAWGzM9NV8Px3UcPLwf1rv9jQptvK33A"

    logs = DatabaseLogger("positions", "sniper")
    logs.qol_upsert_positions()
