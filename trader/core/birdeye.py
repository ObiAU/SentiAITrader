import requests, threading, time, logging, os
from time import sleep
from typing import List, Optional, Any, Union, Dict, Tuple
from pydantic import BaseModel, HttpUrl, field_validator
from datetime import datetime, timezone, timedelta
from pydantic.error_wrappers import ValidationError
from jupiter import verify_tx_and_get_balance_change
from solana.rpc.api import Client
from trader.config import Config


class BirdEyeConfig(BaseModel):
    api_key: str
    base_url: HttpUrl = "https://public-api.birdeye.so"

class Candle(BaseModel):
    """A Pydantic model for parsing OHLCV data."""
    o: float       # open
    h: float       # high
    l: float       # low
    c: float       # close
    v: float       # volume
    unixTime: int
    address: str
    type: str


class TokenInfo(BaseModel):
    address: str
    name: Optional[str] = None
    symbol: Optional[str] = None
    decimals: int
    balance: str
    uiAmount: float
    chainId: str
    logoURI: Optional[str] = None # to encompass both HttpUrl and str
    priceUsd: Optional[float] = None
    valueUsd: Optional[float] = None

    @field_validator("balance", mode="before")
    def ensure_string_balance(cls, value):
        return str(value) if isinstance(value, (int, float)) else value

class Transaction(BaseModel):
    txHash: str
    blockNumber: int
    blockTime: str
    status: bool
    from_address: Optional[str] = None
    to_address: Optional[str] = None
    fee: Optional[int] = None
    mainAction: Optional[str] = None
    balanceChange: Optional[List[dict]] = None
    contractLabel: Optional[dict] = None

class Trader(BaseModel):
    network: Optional[str] = None
    address: Optional[str] = None
    pnl: Optional[float] = 0.0
    trade_count: Optional[int] = 0
    volume: Optional[float] = 0.0

    class Config:
        extra = "allow"  # Ignore extra fields

class TokenSecurityInfo(BaseModel):
    creatorAddress: Optional[str]
    creatorOwnerAddress: Optional[str]
    ownerAddress: Optional[str]
    ownerOfOwnerAddress: Optional[str]
    creationTx: Optional[str]
    creationTime: Optional[int]
    creationSlot: Optional[int]
    mintTx: Optional[str]
    mintTime: Optional[int]
    mintSlot: Optional[int]
    creatorBalance: Optional[float]
    ownerBalance: Optional[float]
    ownerPercentage: Optional[float]
    creatorPercentage: Optional[float]
    metaplexUpdateAuthority: Optional[str]
    metaplexOwnerUpdateAuthority: Optional[str]
    metaplexUpdateAuthorityBalance: Optional[float]
    metaplexUpdateAuthorityPercent: Optional[float]
    mutableMetadata: Optional[bool]
    top10HolderBalance: Optional[float]
    top10HolderPercent: Optional[float]
    top10UserBalance: Optional[float]
    top10UserPercent: Optional[float]
    isTrueToken: Optional[bool]
    totalSupply: Optional[float]
    preMarketHolder: List[Any]
    lockInfo: Optional[Any]
    freezeable: Optional[bool]
    freezeAuthority: Optional[str]
    transferFeeEnable: Optional[bool]
    transferFeeData: Optional[Any]
    isToken2022: Optional[bool]
    nonTransferable: Optional[bool]

    @property
    def creation_time_human(self) -> Optional[str]:
        """Convert creationTime from UNIX timestamp to a human-readable format."""
        return datetime.utcfromtimestamp(self.creationTime).strftime('%Y-%m-%d %H:%M:%S') if self.creationTime else None

    @property
    def mint_time_human(self) -> Optional[str]:
        """Convert mintTime from UNIX timestamp to a human-readable format."""
        return datetime.utcfromtimestamp(self.mintTime).strftime('%Y-%m-%d %H:%M:%S') if self.mintTime else None

class Extensions(BaseModel):
    coingeckoId: Optional[str] = None
    serumV3Usdc: Optional[str] = None
    serumV3Usdt: Optional[str] = None
    website: Optional[HttpUrl] = None
    telegram: Optional[str] = None
    twitter: Optional[HttpUrl] = None
    description: Optional[str] = None
    discord: Optional[HttpUrl] = None
    medium: Optional[HttpUrl] = None

class TokenOverview(BaseModel):
    address: Optional[str]
    decimals: Optional[int]
    symbol: Optional[str]
    name: Optional[str]
    extensions: Optional[Extensions]
    logoURI: Optional[str] = None
    liquidity: Optional[float] = 0.0
    lastTradeUnixTime: Optional[int]
    lastTradeHumanTime: Optional[str]
    price: Optional[float] = 0.0
    history30mPrice: Optional[float]
    priceChange30mPercent: Optional[float]
    history1hPrice: Optional[float]
    priceChange1hPercent: Optional[float]
    history2hPrice: Optional[float]
    priceChange2hPercent: Optional[float]
    history4hPrice: Optional[float]
    priceChange4hPercent: Optional[float]
    history6hPrice: Optional[float]
    priceChange6hPercent: Optional[float]
    history8hPrice: Optional[float]
    priceChange8hPercent: Optional[float]
    history12hPrice: Optional[float]
    priceChange12hPercent: Optional[float]
    history24hPrice: Optional[float]
    priceChange24hPercent: Optional[float]
    uniqueWallet30m: Optional[int]
    uniqueWalletHistory30m: Optional[int]
    uniqueWallet30mChangePercent: Optional[float]
    uniqueWallet1h: Optional[int]
    uniqueWalletHistory1h: Optional[int]
    uniqueWallet1hChangePercent: Optional[float]
    uniqueWallet2h: Optional[int]
    uniqueWalletHistory2h: Optional[int]
    uniqueWallet2hChangePercent: Optional[float]
    uniqueWallet4h: Optional[int]
    uniqueWalletHistory4h: Optional[int]
    uniqueWallet4hChangePercent: Optional[float]
    uniqueWallet8h: Optional[int]
    uniqueWalletHistory8h: Optional[int]
    uniqueWallet8hChangePercent: Optional[float]
    uniqueWallet24h: Optional[int]
    uniqueWalletHistory24h: Optional[int]
    uniqueWallet24hChangePercent: Optional[float]
    supply: Optional[float] = 0.0
    mc: Optional[float] = 0.0
    circulatingSupply: Optional[float] = 0.0
    realMc: Optional[float] = 0.0
    holder: Optional[int]
    trade30m: Optional[int]
    tradeHistory30m: Optional[int]
    trade30mChangePercent: Optional[float]
    sell30m: Optional[int]
    sellHistory30m: Optional[int]
    sell30mChangePercent: Optional[float]
    buy30m: Optional[int]
    buyHistory30m: Optional[int]
    buy30mChangePercent: Optional[float]
    v30m: Optional[float]
    v30mUSD: Optional[float]
    vHistory30m: Optional[float]
    vHistory30mUSD: Optional[float]
    v30mChangePercent: Optional[float]
    vBuy30m: Optional[float]
    vBuy30mUSD: Optional[float]
    vBuyHistory30m: Optional[float]
    vBuyHistory30mUSD: Optional[float]
    vBuy30mChangePercent: Optional[float]
    vSell30m: Optional[float]
    vSell30mUSD: Optional[float]
    vSellHistory30m: Optional[float]
    vSellHistory30mUSD: Optional[float]
    vSell30mChangePercent: Optional[float]
    trade1h: Optional[int]
    tradeHistory1h: Optional[int]
    trade1hChangePercent: Optional[float]
    sell1h: Optional[int]
    sellHistory1h: Optional[int]
    sell1hChangePercent: Optional[float]
    buy1h: Optional[int]
    buyHistory1h: Optional[int]
    buy1hChangePercent: Optional[float]
    v1h: Optional[float]
    v1hUSD: Optional[float]
    vHistory1h: Optional[float]
    vHistory1hUSD: Optional[float]
    v1hChangePercent: Optional[float]
    vBuy1h: Optional[float]
    vBuy1hUSD: Optional[float]
    vBuyHistory1h: Optional[float]
    vBuyHistory1hUSD: Optional[float]
    vBuy1hChangePercent: Optional[float]
    vSell1h: Optional[float]
    vSell1hUSD: Optional[float]
    vSellHistory1h: Optional[float]
    vSellHistory1hUSD: Optional[float]
    vSell1hChangePercent: Optional[float]
    trade2h: Optional[int]
    tradeHistory2h: Optional[int]
    trade2hChangePercent: Optional[float]
    sell2h: Optional[int]
    sellHistory2h: Optional[int]
    sell2hChangePercent: Optional[float]
    buy2h: Optional[int]
    buyHistory2h: Optional[int]
    buy2hChangePercent: Optional[float]
    v2h: Optional[float]
    v2hUSD: Optional[float]
    vHistory2h: Optional[float]
    vHistory2hUSD: Optional[float]
    v2hChangePercent: Optional[float]
    vBuy2h: Optional[float]
    vBuy2hUSD: Optional[float]
    vBuyHistory2h: Optional[float]
    vBuyHistory2hUSD: Optional[float]
    vBuy2hChangePercent: Optional[float]
    vSell2h: Optional[float]
    vSell2hUSD: Optional[float]
    vSellHistory2h: Optional[float]
    vSellHistory2hUSD: Optional[float]
    vSell2hChangePercent: Optional[float]
    trade4h: Optional[int]
    tradeHistory4h: Optional[int]
    trade4hChangePercent: Optional[float]
    sell4h: Optional[int]
    sellHistory4h: Optional[int]
    sell4hChangePercent: Optional[float]
    buy4h: Optional[int]
    buyHistory4h: Optional[int]
    buy4hChangePercent: Optional[float]
    v4h: Optional[float]
    v4hUSD: Optional[float]
    vHistory4h: Optional[float]
    vHistory4hUSD: Optional[float]
    v4hChangePercent: Optional[float]
    vBuy4h: Optional[float]
    vBuy4hUSD: Optional[float]
    vBuyHistory4h: Optional[float]
    vBuyHistory4hUSD: Optional[float]
    vBuy4hChangePercent: Optional[float]
    vSell4h: Optional[float]
    vSell4hUSD: Optional[float]
    vSellHistory4h: Optional[float]
    vSellHistory4hUSD: Optional[float]
    vSell4hChangePercent: Optional[float]
    trade8h: Optional[int]
    tradeHistory8h: Optional[int]
    trade8hChangePercent: Optional[float]
    sell8h: Optional[int]
    sellHistory8h: Optional[int]
    sell8hChangePercent: Optional[float]
    buy8h: Optional[int]
    buyHistory8h: Optional[int]
    buy8hChangePercent: Optional[float]
    v8h: Optional[float]
    v8hUSD: Optional[float]
    vHistory8h: Optional[float]
    vHistory8hUSD: Optional[float]
    v8hChangePercent: Optional[float]
    vBuy8h: Optional[float]
    vBuy8hUSD: Optional[float]
    vBuyHistory8h: Optional[float]
    vBuyHistory8hUSD: Optional[float]
    vBuy8hChangePercent: Optional[float]
    vSell8h: Optional[float]
    vSell8hUSD: Optional[float]
    vSellHistory8h: Optional[float]
    vSellHistory8hUSD: Optional[float]
    vSell8hChangePercent: Optional[float]
    trade24h: Optional[int]
    tradeHistory24h: Optional[int]
    trade24hChangePercent: Optional[float]
    sell24h: Optional[int]
    sellHistory24h: Optional[int]
    sell24hChangePercent: Optional[float]
    buy24h: Optional[int]
    buyHistory24h: Optional[int]
    buy24hChangePercent: Optional[float]
    v24h: Optional[float]
    v24hUSD: Optional[float]
    vHistory24h: Optional[float]
    vHistory24hUSD: Optional[float]
    v24hChangePercent: Optional[float]
    vBuy24h: Optional[float]
    vBuy24hUSD: Optional[float]
    vBuyHistory24h: Optional[float]
    vBuyHistory24hUSD: Optional[float]
    vBuy24hChangePercent: Optional[float]
    vSell24h: Optional[float]
    vSell24hUSD: Optional[float]
    vSellHistory24h: Optional[float]
    vSellHistory24hUSD: Optional[float]
    vSell24hChangePercent: Optional[float]
    watch: Optional[Union[str, int]] = None  # Allow both string and integer types
    numberMarkets: Optional[int]


class BirdEyeClient:
    def __init__(self, config: BirdEyeConfig):
        self.config = config

        # Thread-safe rate limiting variables
        self._lock = threading.Lock()
        self._max_requests_per_second = 10
        self._min_interval = 1 / self._max_requests_per_second
        self._last_request_time = 0.0
        self.solana_client = Client(Config.RPC_URL)

    def _create_url(self, endpoint: str, **params) -> str:
        """Generate the URL with query parameters."""
        query_string = "&".join([f"{key}={value}" for key, value in params.items() if value is not None])
        return f"{self.config.base_url}/{endpoint}?{query_string}" if params else f"{self.config.base_url}/{endpoint}"

    def _headers(self) -> dict:
        return {
            "accept": "application/json",
            "x-chain": "solana",
            "X-API-KEY": self.config.api_key,
        }

    def _rate_limit(self):
        """
        Enforce thread-safe rate limiting for API calls
        (maximum of 15 requests per second globally).
        """
        with self._lock:
            now = time.perf_counter()
            elapsed = now - self._last_request_time
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last_request_time = time.perf_counter()

    def fetch_balances(self, wallet: str) -> List[TokenInfo]:
        self._rate_limit()
        url = self._create_url("v1/wallet/token_list", wallet=wallet)
        response = requests.get(url, headers=self._headers())
        response.raise_for_status()
        data = response.json()

        return [TokenInfo(**item) for item in data["data"].get("items", [])]

    def get_token_balance(self, wallet_address: str, token_address: str) -> float:
        """
        convenience func for a specific token
        """
        try:
            balances = self.fetch_balances(wallet_address)
            for token in balances:
                if token.address == token_address:
                    token_decimals = token.decimals or 0  # Handle potential None values
                    token_balance = float(token.balance) / (10 ** token_decimals)
                    return token_balance
        except Exception as e:
            print(f"Error fetching balances for wallet {wallet_address}: {e}")

        # Return 0.0 if the token address is not found or an error occurs
        return 0.0
    
    def get_all_balances(self, wallet_address: str) -> float:
        """
        Check balances against positions in database.
        """
        try:
            balances = self.fetch_balances(wallet_address)
            return balances
        except Exception as e:
            print(f"Error fetching balances for wallet {wallet_address}: {e}")

        # Return 0.0 if the token address is not found or an error occurs
        return 0.0


    def get_balance_change_since_tx(self,
        tx_sig: str,
        user_pubkey: str,
        output_mint: str,
        output_decimals: int,
    ) -> Tuple[str, float]:
        """
        Checks how a user's balance in a specific token has changed
        between the transaction's finalized post-balance and the current balance.

        :param solana_client: An initialized Solana `Client` for RPC queries.
        :param bird_eye_client: BirdEye client that can fetch token balances.
        :param tx_sig: The transaction signature to verify.
        :param user_pubkey: The user's public key as a string.
        :param output_mint: The mint address (string) of the token in question.
        :param output_decimals: The token's decimals.
        :return: Tuple of (transaction_status, balance_diff_since_tx).
                `transaction_status` will be one of:
                    - "notFound"
                    - "notFinalized"
                    - "failed"
                    - "finalized"
                `balance_diff_since_tx` is (current_balance - post_tx_balance).
                If transaction is not "finalized", returns 0.0 for the difference.
        """

        # 1) Verify transaction and get post-TX balance
        tx_status, tx_balance_change, post_balance_int = verify_tx_and_get_balance_change(
            client=self.solana_client,
            tx_sig=tx_sig,
            user_pubkey=user_pubkey,
            output_mint=output_mint,
            output_decimals=output_decimals,
        )

        # If transaction is not finalized or failed/not found, return early
        if tx_status != "finalized":
            # If not found/finalized, the difference is effectively 0.0
            return tx_status, 0.0

        # post_balance_int is the raw integer from the transaction logs.
        # Convert it to a float by dividing by 10**decimals:
        post_balance_float = 0.0
        if post_balance_int is not None:
            post_balance_float = post_balance_int / (10 ** output_decimals)

        # 2) Fetch current balance using BirdEye
        current_balance_float = self.get_token_balance(
            wallet_address=user_pubkey,
            token_address=output_mint,
        )

        # 3) Calculate how the balance has changed since the end of that transaction
        balance_diff_since_tx = current_balance_float - post_balance_float

        return tx_status, balance_diff_since_tx


    def fetch_transaction_history(self, wallet: str, limit: int = 10, offset: int = 0) -> List[Transaction]:
        self._rate_limit()
        url = self._create_url("v1/wallet/tx_list", wallet=wallet, limit=limit, offset=offset)
        response = requests.get(url, headers=self._headers())
        response.raise_for_status()
        data = response.json()
        return [Transaction(**item) for item in data["data"].get("solana", [])]

    def extract_tokens_from_transactions(transactions: List[Transaction]) -> set:
        tokens = set()
        for tx in transactions:
            # Use datetime.strptime to parse ISO 8601 format with timezone
            tx_time = datetime.strptime(tx.blockTime, '%Y-%m-%dT%H:%M:%S%z')
            if datetime.now(timezone.utc) - tx_time <= timedelta(minutes=10):
                if tx.balanceChange:
                    for change in tx.balanceChange:
                        if change.get("amount", 0) > 0 and change.get("symbol") != "SOL":  # Exclude native SOL
                            tokens.add(change.get("address"))
        return tokens
    
    def fetch_current_token_balance_safely(self, wallet_pubkey: str, token_mint: str) -> float:

        try:
            balances = self.fetch_balances(wallet_pubkey)
            for tkn in balances:
                if tkn.address == token_mint:
                    return float(tkn.balance) / 10**tkn.decimals
            return 0.0
        except Exception as e:
            logging.error(f"Could not fetch balance for {token_mint}: {e}")
            return 0.0


    def fetch_top_traders(self, count: int = 10, period: str = "1W") -> List[Trader]:
        results = []
        limit = 10
        for offset in range(0, count, limit):
            try:
                self._rate_limit()

                url = self._create_url(
                    "trader/gainers-losers",
                    type=period,
                    sort_by="PnL",
                    sort_type="desc",
                    offset=offset,
                    limit=limit,
                )

                response = requests.get(url, headers=self._headers())

                if not response.text.strip():
                    print(f"Empty response for offset {offset}. Skipping.")
                    continue
                if "application/json" not in response.headers.get("Content-Type", ""):
                    print(f"Unexpected Content-Type for offset {offset}: {response.headers.get('Content-Type')}")
                    continue

                response.raise_for_status()

                data = response.json()
                results.extend([Trader(**item) for item in data["data"].get("items", [])])
            except requests.exceptions.JSONDecodeError:
                print(f"Failed to parse JSON for offset {offset}: {response.text[:500]}")
                continue
            except Exception as e:
                print(f"Error fetching data for offset {offset}: {e}")
                continue
        return results

    def fetch_token_security_info(self, address: str) -> Optional[TokenSecurityInfo]:
        """
        Fetch security information for a given token address.

        Args:
            address (str): The token address to fetch security info for.

        Returns:
            Optional[TokenSecurityInfo]: Parsed security information for the token, or None if the request fails.
        """
        self._rate_limit()
        url = f"https://public-api.birdeye.so/defi/token_security?address={address}"  # Direct URL

        headers = {
            "accept": "application/json",
            "x-chain": "solana",
            "X-API-KEY": self.config.api_key
        }

        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            logging.error(f"Error: Received {response.status_code} for address {address}")
            logging.error(f"Response Text: {response.text}")
            return None  # Gracefully handle errors without raising an exception

        try:
            data = response.json()
            return TokenSecurityInfo(**data["data"])
        except requests.exceptions.JSONDecodeError:
            print(f"Failed to parse JSON response for address {address}: {response.text[:500]}")
            return None


    # def fetch_token_trade_data(self, address: str):

    def fetch_token_overview(self, address: str) -> Optional[TokenOverview]:
        """
        Fetch the overview of a given token.

        Args:
            address (str): The token address to fetch the overview for.

        Returns:
            Optional[TokenOverview]: Parsed token overview data, or None if the request fails.
        """
        self._rate_limit()
        url = f"https://public-api.birdeye.so/defi/token_overview?address={address}"  # Direct URL

        response = requests.get(url, headers=self._headers())

        if response.status_code != 200:
            logging.error(f"Error: Received {response.status_code} for address {address}")
            logging.error(f"Response Text: {response.text}")
            return None  # Gracefully handle errors without raising an exception

        try:
            data = response.json()
            if "data" in data and isinstance(data["data"], dict):
                # Safely instantiate the model with partial data
                return TokenOverview(**{key: data["data"].get(key) for key in TokenOverview.model_fields})
            else:
                logging.error(f"Unexpected response format for address {address}: {data}")
                return None
        except requests.exceptions.JSONDecodeError:
            logging.error(f"Failed to parse JSON response for address {address}: {response.text[:500]}")
            return None
        except ValidationError as e:
            logging.error(f"Validation error for address {address}: {e}")
            return None
    

    def fetch_new_holders_last_24h(self, address: str) -> Optional[List[str]]:
        """
        Fetch the list of new holders who added the token in the last 24 hours.

        Args:
            address (str): The token address.

        Returns:
            Optional[List[str]]: List of unique wallet addresses that added the token in the last 24 hours.
        """
        # unix 24h vs unix now
        before_time = int(time.time()) 
        after_time = int(time.time()) - 24 * 60 * 60
        url = f"https://public-api.birdeye.so/defi/txs/token/seek_by_time"

        params = {
            "address": address,
            "tx_type": "add",
            # "before_time": before_time,
            "after_time": after_time,
            "limit": 50
        }

        response = requests.get(url, headers=self._headers(), params=params)

        if response.status_code != 200:
            logging.error(f"Error: Received {response.status_code} for address {address}")
            logging.error(f"Response Text: {response.text}")
            return None

        try:
            data = response.json()
            # print(f"New Holders Data:\n {data}")
            if data.get("success") and "items" in data.get("data", {}):
                transactions = data["data"]["items"]
                unique_owners_24h = {tx["owner"] for tx in transactions if "owner" in tx}
                return len(unique_owners_24h)
            else:
                logging.error(f"Unexpected response format: {data}")
                return None
        except requests.exceptions.JSONDecodeError:
            logging.error(f"Failed to parse JSON response: {response.text[:500]}")
            return None


    def fetch_candles(
        self,
        address: str,
        timeframe: str = "5m",
        time_from: Optional[int] = None,
        time_to: Optional[int] = None
    ) -> List[Candle]:
        """
        Fetch OHLCV (candlestick) data for a given token within an optional date/time range.

        Args:
            address (str): The token address (mint address on Solana) to fetch OHLCV data for.
            timeframe (str): Desired timeframe (e.g. "5m", "15m", "1h", "4h", "1d").
            time_from (int, optional): UNIX timestamp representing start time for data.
            time_to (int, optional): UNIX timestamp representing end time for data.

        Returns:
            List[Candle]: A list of Candle objects containing OHLCV data.
        """

        # Enforce rate limiting (if you have a _rate_limit method)
        self._rate_limit()

        # Prepare query parameters
        params = {
            "address": address,
            "type": timeframe,
        }
        # expects startTime and endTime in unix timestamp
        if time_from is not None:
            params["time_from"] = time_from
        if time_to is not None:
            params["time_to"] = time_to

        url = self._create_url("defi/ohlcv", **params)
        print(f"Fetching candles via {url}")

        # Make the request
        response = requests.get(url, headers=self._headers())
        response.raise_for_status()

        data = response.json()
        items = data["data"].get("items", [])

        # Convert each dictionary in "items" to a Candle object
        candles = [Candle(**item) for item in items]
        return candles

    def fetch_live_price(self, address: str) -> float:
        """
        Fetch the current (live) price for a given token address.

        Args:
            address (str): The token address (mint address on Solana).

        Returns:
            float: The current price from BirdEye, or raises an exception if the request fails.
        """
        self._rate_limit()
        url = self._create_url("defi/price", address=address)
        response = requests.get(url, headers=self._headers())
        response.raise_for_status()

        data = response.json()
        # "data" should contain keys like {"value": ..., "updateUnixTime": ..., ...}

        # access data["data"]["value"] safely

        value = data.get("data", {}).get("value")

        return value

    def fetch_historical_prices_unix(
        self,
        address: str,
        start_time: int,
        end_time: int,
        candles: str = "1D"
    ) -> Optional[List[Dict[str, float]]]:

        base_url = f"{self.config.base_url}/defi/history_price"
        # base_url = self._create_url("defi/history_price", address=address)

        self._rate_limit()

        params = {
            "address": address,
            "address_type": "token",
            "type": candles,  # try 3D. Do not try 1M
            "time_from": start_time,
            "time_to": end_time
        }
        try:
            resp = requests.get(base_url, headers=self._headers(), params=params, timeout=10)
            data = resp.json()
            # print(f"Data: {data}")
            if data.get("success"):
                items = data["data"]["items"]
                # Sort by unixTime asc
                items.sort(key=lambda x: x["unixTime"])
                return items
            else:
                logging.warning(f"BirdEye error: {data}")
                return None
        except Exception as e:
            logging.error(f"Exception in fetch_historical_prices_unix: {e}")
            return None

def tester():
    import pandas as pd, time
    from ta.trend import ADXIndicator
    config = BirdEyeConfig(api_key=Config.BIRDEYE_API_KEY)
    client = BirdEyeClient(config=config)
    # WRAPPED_SOL_MINT_ADDRESS = "So11111111111111111111111111111111111111111"
    WRAPPED_SOL_OHLCV_ADDRESS = "So11111111111111111111111111111111111111112"

    def update_market_trend(window: int = 10, adx_threshold: float = 23):
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
            candles = client.fetch_candles(
                address=WRAPPED_SOL_OHLCV_ADDRESS,
                timeframe="30m",
                time_from=time_from,
                time_to=now
            )
        except Exception as e:
            logging.error(f"Error fetching SOL candles for market trend analysis: {e}")
            market_trend = "none"
            return

        if not candles:
            logging.warning("No SOL candle data available for market trend analysis.")
            market_trend = "none"
            return

        df = pd.DataFrame([c.model_dump() for c in candles]) # fetch candles gives list of Candle objects

        if len(df) < window:
            logging.warning("Not enough candle data to compute ADX. Setting market trend to 'none'.")
            market_trend = "none"
            return
        try:
            df['high'] = df['h'].astype(float)
            df['low'] = df['l'].astype(float)
            df['close'] = df['c'].astype(float)
        except Exception as e:
            logging.error(f"Error processing candle data for market trend analysis: {e}")
            market_trend = "none"
            return

        # Calculate ADX and its directional indices.
        adx_indicator = ADXIndicator(
            high=df['high'], 
            low=df['low'], 
            close=df['close'], 
            window=window
            )
        print(f"ADX indicator: {adx_indicator}")
        try:
            adx = adx_indicator.adx().iloc[-1]
            print(f"ADX: {adx}")
            di_plus = adx_indicator.adx_pos().iloc[-1]
            print(f"DI+: {di_plus}")
            di_minus = adx_indicator.adx_neg().iloc[-1]
            print(f"DI-: {di_minus}")
        except Exception as e:
            logging.error(f"Error calculating ADX for market trend: {e}")
            market_trend = "none"
            return

        logging.info(f"Market Trend Analysis - ADX: {adx:.2f}, DI+: {di_plus:.2f}, DI-: {di_minus:.2f} at {datetime.now()}")

        if di_minus > di_plus and adx > adx_threshold:
            market_trend = "negative"
        elif di_plus > di_minus and adx > adx_threshold:
            market_trend = "positive"
        else:
            market_trend = "none"

        print(market_trend)
    
    update_market_trend()



if __name__ == "__main__":
    tester()

