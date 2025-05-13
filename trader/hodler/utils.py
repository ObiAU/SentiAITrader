import os, sys, logging, json, pandas as pd, numpy as np
from datetime import datetime, timedelta
from typing import Tuple

## passing from wallet / cache to sentiment agent. Need to pass both ticker symbol and token name.
# e.g. reddit search will search both ticker and token name. Also can attempt subreddit search of token name (see what comes up)

CACHE_FILE = "hodl_checks.json"
HODLERS_FILE = "big_hodlers.json"

def detailed_sell_strategy(
    sentiment: float,
    cultiness: float,
    entry_price: float,
    current_price: float,
    max_price: float,
    partial_sold: float,
    original_amount: float,
) -> Tuple[float, str]:
    """
    Returns (fraction_to_sell, reason) based on sentiment, cultiness, and price performance.

    - fraction_to_sell: 0.0 to 1.0 (how much of the remaining position to sell).
    - reason: Explanation for logging.
    """
    # === Sentiment-based logic ===
    if sentiment < 0.20:
        return 1.0, "Sentiment < 0.20 (Full Exit)"
    elif sentiment < 0.40:
        fraction_to_sell = 0.25
        reason = "Sentiment < 0.40 (Partial Exit)"
    elif sentiment < 0.50:
        fraction_to_sell = 0.10
        reason = "Sentiment < 0.50 (Minor Risk Reduction)"
    else:
        fraction_to_sell = 0.0
        reason = ""

    # Adjust for cultiness
    if cultiness >= 0.7:
        fraction_to_sell *= 0.5  # Reduce sell fraction for high cultiness
        reason += " | Cultiness >= 0.7 (Reduced Sell)"
    elif cultiness < 0.3:
        fraction_to_sell *= 1.5  # Increase sell fraction for low cultiness
        reason += " | Cultiness < 0.3 (Increased Sell)"

    # === Price-based logic ===
    current_factor = current_price / entry_price if entry_price > 0 else 0
    price_based_fraction = 0.0

    partial_sells_table = [
        (2.0, 0.25),  # At 2×, sell 25% of the original position
        (3.0, 0.50),  # At 3×, sell 50% total
        (5.0, 1.00),  # At 5×, sell 100%
    ]

    for (threshold, target_fraction) in partial_sells_table:
        if current_factor >= threshold and partial_sold < target_fraction:
            price_based_fraction += target_fraction - partial_sold
            reason += f" | Price threshold {threshold}× hit"

    # === Trailing stop logic ===
    trailing_stop_factor = 0.5 if sentiment >= 0.6 else 0.3  # Adjust stop-loss based on sentiment
    stop_price = max_price * trailing_stop_factor

    if current_price < stop_price:
        price_based_fraction = 1.0  # Full exit
        reason += f" | Trailing stop triggered at {stop_price:.2f} USDC"

    # Combine signals (use max of sentiment or price-based fractions)
    fraction_to_sell = max(fraction_to_sell, price_based_fraction)

    # Final adjustments (clamp between 0.0 and 1.0)
    fraction_to_sell = max(0.0, min(fraction_to_sell, 1.0))

    return fraction_to_sell, reason


def get_top_tokens_for_wallet(wallet_address):
    """
    get top tokens from a given wallet address. Pass marketcap checks though
    """
    # Dummy placeholder
    return [
        {"address": f"{wallet_address}-token1", "symbol": "TKN1"},
        {"address": f"{wallet_address}-token2", "symbol": "TKN2"}
    ]

# def murad_strategy(token_address)


def scout_tokens(max_cache_age_days=15) -> list:
    """
    Check if tokens are cached in `hodl_checks.json` and if they are newer than
    `max_cache_age_days`. If not, refresh the cache by reading addresses from
    `big_hodlers.json`, retrieving top tokens by marketcap criteria, and storing
    (caching) them in `hodl_checks.json` with a new scout_time.
    """
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            cache_data = json.load(f)

        scout_time_str = cache_data.get("scout_time")
        cached_tokens = cache_data.get("tokens", [])

        if scout_time_str:
            scout_time = datetime.fromisoformat(scout_time_str)
            if datetime.now() - scout_time < timedelta(days=max_cache_age_days):
                print("Using cached tokens (cache is still valid).")
                return cached_tokens
            else:
                print("Cache is older than 15 days. Refreshing.")
        else:
            print("No scout_time found in cache. Refreshing.")
    else:
        print("No cache file found. Generating new cache.")


    if not os.path.exists(HODLERS_FILE):
        print(f"Could not find file {HODLERS_FILE}.")
        return []

    with open(HODLERS_FILE, "r") as f:
        try:
            hodlers_data = json.load(f) # wallet idx, wallet address
            wallet_addresses = list(hodlers_data.values())

        except json.JSONDecodeError as e:
            print(f"Error parsing big_hodlers.json. Error: {e}")
            return []

    new_token_list = []
    for wallet_address in wallet_addresses:
        wallet_tokens = get_top_tokens_for_wallet(wallet_address)
        new_token_list.extend(wallet_tokens)

    # de-duplicate
    unique_tokens = {}
    for token in new_token_list:
        unique_tokens[token['address']] = token
    new_token_list = list(unique_tokens.values())

    cache_data = {
        "scout_time": datetime.now().isoformat(),
        "tokens": new_token_list
    }
    with open(CACHE_FILE, "w") as f:
        json.dump(cache_data, f, indent=2)

    print("Refreshed and cached new token list.")
    return new_token_list


if __name__ == "__main__":
    tokens = scout_tokens(max_cache_age_days=15)
    print(f"Found {len(tokens)} tokens.")
