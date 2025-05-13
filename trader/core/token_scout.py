import asyncio, sys, os, random, logging
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Set, Tuple, Optional, Dict, Any

from birdeye import BirdEyeClient, BirdEyeConfig, TokenSecurityInfo, TokenOverview, Trader
from trader.core.dexscreener import fetch_dexscreener_info
from trader.core.top_traders import top_trader_addresses
from trader.sniper.analysis import analyze_for_good_tokens
from trader.core.models import TokenMetadata
from trader.config import Config

# ---------------------------------------------------------
# GLOBAL BirdEyeConfig (not a shared client)
# ---------------------------------------------------------
global_config = BirdEyeConfig(api_key=Config.BIRDEYE_API_KEY)
import time

# Retained traders across iterations
retained_traders: List[str] = []

ignore_tokens = ["EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v", "So11111111111111111111111111111111111111112", "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"] # USDC, SOL, USDT


# ---------------------------------------------------------
# ASYNC transaction fetch
# ---------------------------------------------------------
async def fetch_transactions_async(trader: str, trade_opened_less_than_x_mins_ago: Optional[int] = None) -> tuple[set, int]:
    """
    Asynchronously fetch transactions for `trader` in a thread pool,
    returning (valid_addresses, valid_count).
    We create a new BirdEyeClient *inside* the function so each task
    has its own instance (helps avoid thread-safety / rate-limit issues).
    """
    def sync_fetch():
        try:
            local_client = BirdEyeClient(config=global_config)
            return local_client.fetch_transaction_history(wallet=trader, limit=50)
        except Exception as e:
            # Re-raise the exception to be caught by asyncio.gather
            raise e

    # Offload blocking calls to a default thread pool executor:
    loop = asyncio.get_running_loop()
    transactions = await loop.run_in_executor(None, sync_fetch)

    valid_addresses = set()
    tx_details = []

    for tx in transactions:
        if tx.mainAction == "received" or tx.mainAction == "send":
            continue

        tx_time = datetime.fromisoformat(tx.blockTime.replace("Z", "+00:00"))

        if trade_opened_less_than_x_mins_ago:
            if datetime.now(timezone.utc) - tx_time <= timedelta(minutes=trade_opened_less_than_x_mins_ago):
                if tx.balanceChange:
                    for change in tx.balanceChange:
                        if change.get("amount", 0) > 0 and change.get("symbol") not in ["SOL", "USDC"] and change.get("address") not in ignore_tokens:
                            valid_addresses.add(change.get("address"))
                            tx_details.append({
                                "token_address": change.get("address"),
                                "trader": trader,
                                "tx_time": tx_time,
                                "tx_hash": tx.txHash
                            })
        
        # for sentitrader to pickup hodling positions
        else:
            if tx.balanceChange:
                for change in tx.balanceChange:
                    if change.get("amount", 0) > 0 and change.get("symbol") not in ["SOL", "USDC"] and change.get("address") not in ignore_tokens:
                        valid_addresses.add(change.get("address"))
                        tx_details.append({
                            "token_address": change.get("address"),
                            "trader": trader,
                            "tx_time": tx_time,
                            "tx_hash": tx.txHash
                        })

    return valid_addresses, tx_details

# ---------------------------------------------------------
# REMAINING STEPS (security info, dex, overview, etc.)
# ---------------------------------------------------------

config_client = BirdEyeClient(config=global_config)  # Shared client for the rest


def fetch_security_info(address):
    """
    Synchronously fetch token security info using the global (shared) config_client.
    """
    return address, config_client.fetch_token_security_info(address)


def fetch_token_overview(address):
    """
    Synchronously fetch token overview using the global (shared) config_client.
    """
    return address, config_client.fetch_token_overview(address)

def fetch_balance_change_since_tx(
        tx_sig: str,
        user_pubkey: str,
        output_mint: str,
        output_decimals: int) -> float:

    """
    Fetch the balance change since a given transaction for a given user and token.
    """

    return config_client.get_balance_change_since_tx(tx_sig, user_pubkey, output_mint, output_decimals)

def fetch_token_overview_and_metadata(address):

    overview = config_client.fetch_token_overview(address)
    new_holders = config_client.fetch_new_holders_last_24h(address)

    symbol = None
    name = None
    holders = None
    market_cap = None
    holders_to_market_cap_ratio = None
    volume_to_market_cap_ratio = None
    holders_increase_percent = None
    trading_volume_24h = None
    buy_volume_24h = None
    sell_volume_24h = None
    buy_sell_ratio_24h = None
    website = twitter = desc = logo_URI = discord = telegram = None
    links = []

    if overview:
        try:
            if hasattr(overview, "symbol") and overview.symbol:
                symbol = overview.symbol
            if hasattr(overview, "name") and overview.name:
                name = overview.name
            if hasattr(overview, "holder") and overview.holder:
                holders = overview.holder
            if hasattr(overview, "mc") and overview.mc:
                market_cap = overview.mc
            if hasattr(overview, "logoURI") and overview.logoURI:
                logo_URI = overview.logoURI
            if hasattr(overview, "v24hUSD") and overview.v24hUSD:
                trading_volume_24h = overview.v24hUSD
            if hasattr(overview, "vBuy24hUSD") and overview.vBuy24hUSD:
                buy_volume_24h = overview.vBuy24hUSD
            if hasattr(overview, "vSell24hUSD") and overview.vSell24hUSD:
                sell_volume_24h = overview.vSell24hUSD

            try:
                if buy_volume_24h is not None and sell_volume_24h is not None:
                    buy_sell_ratio_24h = buy_volume_24h / sell_volume_24h if sell_volume_24h != 0 else float('inf')
            except Exception as e:
                logging.info(f"Error calculating buy_sell_ratio: {e}")

            try:
                if holders and market_cap:
                    holders_to_market_cap_ratio = holders / market_cap
            except Exception as e:
                logging.info(f"Error calculating holders_to_market_cap_ratio: {e}")

            try:
                if trading_volume_24h and market_cap:
                    volume_to_market_cap_ratio = trading_volume_24h / market_cap
            except Exception as e:
                logging.info(f"Error calculating volume_to_market_cap_ratio: {e}")

            try:
                if new_holders and holders:
                    holders_increase_percent = 100 * (new_holders / holders) if holders != 0 else None
            except Exception as e:
                logging.info(f"Error calculating holders_increase_percent: {e}")

            if hasattr(overview, "extensions"):
                website = getattr(overview.extensions, "website", None)
                twitter = getattr(overview.extensions, "twitter", None)
                desc = getattr(overview.extensions, "description", None)
                discord = getattr(overview.extensions, "discord", None)
                telegram = getattr(overview.extensions, "telegram", None)

        except Exception as e:
            logging.info(f"Error processing overview data: {e}")
    
    links = [str(twitter), str(telegram), str(discord), str(website)]

    return {
        "address": address,
        "ticker": symbol,
        "name": name,
        "holders": holders,
        "holders_increase_ratio": holders_increase_percent,
        "holders_to_market_cap_ratio": holders_to_market_cap_ratio,
        "volume_to_market_cap_ratio": volume_to_market_cap_ratio,
        "market_cap": market_cap,
        "volume_24h": trading_volume_24h,
        "buy_volume_24h": buy_volume_24h,
        "sell_volume_24h": sell_volume_24h,
        "buy_sell_ratio_24h": buy_sell_ratio_24h,
        "website": website,
        "twitter": twitter,
        "description": desc,
        "logoURI": logo_URI,
        "urls": links,
        "overview": overview
    }




def scout_tokens(
    *, # make others keyword_args
    mc_bounds: Tuple[float, float] = (50_000, 20_000_000),
    min_trades_in_30m: int = 30,
    max_drawdown_allowed: float = 70.0,
    top10_holder_percent_limit: float = 0.30,
    creation_max_days_ago: int = None,
    creation_min_days_ago: int = 90,   # token must be created within these many days
    creation_max_hours_ago: int = 2,   # token must NOT be created too recently, e.g. last 2 hours
    trade_opened_less_than_x_mins_ago: int = 60, # trade must be opened within these many minutes
    random_sample_size: Optional[int] = None,
    retention_probability: float = 0.95,
    bot_type: str = None
    ):

    global retained_traders
    start_time = datetime.now()

    # --------------------------------------------------------
    # Step 1: Random sampling of traders
    # --------------------------------------------------------

    if random_sample_size:
        if retained_traders:
            # Keep the previously retained traders + sample up to `random_sample_size - len(retained_traders)`
            active_traders = retained_traders + random.sample(
                [t for t in top_trader_addresses if t not in retained_traders],
                k=max(random_sample_size - len(retained_traders), 0)
            )
        else:
            active_traders = random.sample(top_trader_addresses, random_sample_size)
    
    else:
        active_traders = top_trader_addresses
    # --------------------------------------------------------
    # Step 2: ASYNC fetching of transactions
    # --------------------------------------------------------
    async def gather_transactions(traders: List[str]):
        tasks = [fetch_transactions_async(trader, trade_opened_less_than_x_mins_ago) for trader in traders]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        transaction_results = loop.run_until_complete(gather_transactions(active_traders))
    finally:
        loop.close()

    # --------------------------------------------------------
    # Step 3: Combine valid traders and tokens
    # --------------------------------------------------------
    token_addresses = set()
    retained_traders = []  # Reset retained traders for this round
    token_transactions = []

    for idx, result in enumerate(transaction_results):
        trader = active_traders[idx]

        if isinstance(result, Exception):
            print(f"Error fetching transactions for trader {trader}: {result}")
            continue
        
        valid_addresses, tx_details = result
        if tx_details:
            token_transactions.extend(tx_details)

        if valid_addresses and len(valid_addresses) > 0:
            token_addresses.update(valid_addresses)
            # Decide whether we re-retain trader for next round
            if random.random() < retention_probability:
                retained_traders.append(trader)

    print(f"Unique tokens collected: {len(token_addresses)}")

    # --------------------------------------------------------
    # Step 4: Filter tokens by social info
    # --------------------------------------------------------
    social_tokens = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(fetch_dexscreener_info, address): address
            for address in token_addresses
        }
        for future in as_completed(futures):
            address = futures[future]
            try:
                dex_data = future.result()

                if dex_data["has_website"] or dex_data["has_twitter"] or dex_data["has_telegram"]:
                    social_tokens.append(address)
                else:
                    print(
                       f"Skipped token {address} (no website/Twitter/Telegram). "
                       f"(Found: website={dex_data['has_website']}, "
                       f"twitter={dex_data['has_twitter']}, "
                       f"telegram={dex_data['has_telegram']})"
                    )
                    pass
            except Exception as e:
                print(f"Error checking DexScreener for token {address}: {e}")

    # --------------------------------------------------------
    # Step 5: Filter tokens by security info
    # --------------------------------------------------------
    security_info_map = {}
    filtered_tokens = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(fetch_security_info, address): address
            for address in social_tokens
        }
        for future in as_completed(futures):
            address = futures[future]
            try:
                address, security_info = future.result()
                if not security_info:
                    # print(f"Skipped token {address} due to missing security info.")
                    continue

                    # filter on top10 holder % limit
                if (not security_info.top10HolderPercent or
                        security_info.top10HolderPercent > top10_holder_percent_limit):
                    continue

                creation_time = security_info.creationTime

                if creation_time:
                    creation_datetime = datetime.fromtimestamp(creation_time, tz=timezone.utc)

                    print(f"Token {address} creation time: {creation_datetime}")

                    current_time = datetime.now(timezone.utc)

                    min_allowed_creation = current_time - timedelta(days=creation_min_days_ago)
                    max_allowed_creation = current_time - timedelta(hours=creation_max_hours_ago)

                    if creation_max_days_ago:
                        max_allowed_creation = current_time - timedelta(days=creation_max_days_ago)

                    # Must be within [current_time - X days, current_time - Y hours]
                    if not (min_allowed_creation <= creation_datetime <= max_allowed_creation):
                        print(f"Skipped token {address} due to creation time.")
                        continue
                    print(f"Token {address} passed creation time check.")
                else:
                    continue

                if security_info.mutableMetadata:
                    continue

                # If it passes all filters, keep it
                filtered_tokens.append(address)
                security_info_map[address] = security_info

            except Exception as e:
                print(f"Error fetching security info for token {address}: {e}")

    print(f"Tokens remaining after security filters: {len(filtered_tokens)}")

    # --------------------------------------------------------
    # Step 6: Fetch token overview and apply additional filters
    # --------------------------------------------------------
    # Step 6: Fetch token overview and apply additional filters

    final_preanalysis_tokens: List[TokenMetadata] = []

    # --------------------------------------------------
    # 6a) Fetch overviews in parallel
    # --------------------------------------------------
    filtered_overviews = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(fetch_token_overview, address): address
            for address in filtered_tokens
        }
        for future in as_completed(futures):
            address = futures[future]
            try:
                address, overview = future.result()

                # If fetch failed or overview is None, skip
                if not overview:
                    continue

                # Market cap checks
                min_mc, max_mc = mc_bounds
                if not (overview.mc and overview.realMc):
                    continue
                if not (overview.mc >= min_mc and overview.realMc >= min_mc):
                    continue
                if (overview.mc > max_mc and overview.realMc > max_mc):
                    continue

                # Must have at least `min_trades_in_30m` trades in the last 30m
                if not overview.trade30m or overview.trade30m < min_trades_in_30m:
                    continue

                # If it passes all these checks, store it for the next step
                filtered_overviews.append((address, overview))

            except Exception as e:
                print(f"Error fetching token overview for token {address}: {e}")

    # --------------------------------------------------
    # 6b) Perform balance-change checks sequentially
    # --------------------------------------------------
    for address, overview in filtered_overviews:
        matching_transactions = [
            tx for tx in token_transactions if tx["token_address"] == address
        ]

        if len(matching_transactions) == 0:
            continue

        # Take the earliest or relevant transaction
        tx = matching_transactions[0]
        trader = tx["trader"]
        tx_hash = tx["tx_hash"]

        # Sleep 5 seconds to respect rate limits
        time.sleep(5)

        try:
            # Check the trader has not sold the token since buying
            status, balance_change = fetch_balance_change_since_tx(
                tx_sig=tx_hash,
                user_pubkey=trader,
                output_mint=address,
                output_decimals=overview.decimals
            )

            if status != "finalized":
                print(f"Error fetching balance change for token {address}: {status}")
                continue

            if balance_change < 0:
                print(f"Trader {trader} sold some of token {address} since buying. Skipping.")
                continue

            # If still holding, prepare the metadata object
            token_metadata = TokenMetadata(
                address=address,
                mc=overview.mc,
                realMc=overview.realMc,
                decimals=overview.decimals,
                trader=tx["trader"],
                tx_time=tx["tx_time"],
                tx_hash=tx["tx_hash"],
                creation_time=security_info_map[address].creationTime
            )

            final_preanalysis_tokens.append(token_metadata)

        except Exception as e:
            print(f"Error fetching balance change for token {address}: {e}")

    print(f"Tokens passing all pre-analysis filters: {len(final_preanalysis_tokens)}")



    final_analyzed_tokens = final_preanalysis_tokens
    # --------------------------------------------------------
    # Print final results - NB You must run the analysis in the bot code loop to determine a good entry
    # --------------------------------------------------------
    print(f"\n=== Final Tokens After ALL Filters & Analysis: {len(final_analyzed_tokens)} ===")
    # for t in final_analyzed_tokens:
    #     print(f"\nToken Address: {t.address} | Market Cap: {t.mc}")
    #     print("Analysis Results:")
    #     for k, v in t.analysis_results.items():
    #         print(f"   {k}: {v}")

    print(f"\nTotal Time Taken: {datetime.now() - start_time}")

    return final_analyzed_tokens

# ---------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------
if __name__ == "__main__":
    # scout_tokens()
    res = fetch_token_overview_and_metadata("ED5nyyWEzpPPiWimP8vYm7sD7TD3LAt3Q3gRTWHzPJBY") ## moodeng addr
    # print(res.address)
    # print(res.name)
    # print(res.symbol)
    print(res)
