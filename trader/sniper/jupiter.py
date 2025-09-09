import requests, base64, json, threading
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.transaction import VersionedTransaction
from solders.signature import Signature
from solders import message
from solana.rpc.api import Client
from solana.rpc.types import TxOpts
from solana.rpc.commitment import Processed, Finalized
import base58
import time
from solana.rpc.async_api import AsyncClient
from solana.rpc.core import RPCException
import logging
from typing import Tuple, Optional
import asyncio
from trader.config import Config


def update_holdings_from_transaction(
    client: Client,
    tx_sig: str,
    user_pubkey: str,
    holdings: dict,
) -> None:
    """
    1) Fetches the transaction via client.get_transaction(tx_sig).
    2) Checks blockTime and pre/post token balances.
    3) For each changed token, if the transaction's blockTime is newer than
       holdings[mint]["last_update_time"], update local holdings accordingly
       and set holdings[mint]["last_update_time"] to that blockTime.
    """
    resp = client.get_transaction(tx_sig, max_supported_transaction_version=2)
    if not resp.value:
        logging.warning(f"Transaction {tx_sig} not found or not finalized.")
        return

    block_time = resp.value.block_time
    if block_time is None:
        logging.warning(f"No blockTime for tx {tx_sig}.")
        return

    meta = resp.value.transaction.meta
    if not meta:
        logging.warning(f"No meta found for tx {tx_sig}.")
        return

    pre_balances = meta.pre_token_balances or []
    post_balances = meta.post_token_balances or []

    # pre balance map [mint, owner] dict for lookup
    pre_balance_map = {}
    for pre in pre_balances:
        if not pre.mint or not pre.owner:
            continue
        pre_balance_map[(str(pre.mint), str(pre.owner))] = int(pre.ui_token_amount.amount)

    # post
    post_balance_map = {}
    for post in post_balances:
        if not post.mint or not post.owner:
            continue
        post_balance_map[(str(post.mint), str(post.owner))] = int(post.ui_token_amount.amount)

    # check which tokens changed
    changed_mints = set()
    for (mint, owner) in set(pre_balance_map.keys()) | set(post_balance_map.keys()):
        if owner == user_pubkey:
            changed_mints.add(mint)

    # see if block_time is newer for changed mints
    for mint in changed_mints:
        local_info = holdings.get(mint)
        if not local_info:
            # we might not track it yet
            holdings[mint] = {
                "balance": 0.0,
                "decimals": 0,
                "last_update_time": 0,
            }
            local_info = holdings[mint]

        local_time = local_info.get("last_update_time", 0)
        if block_time > local_time:

            pre_amt = pre_balance_map.get((mint, user_pubkey), 0)
            post_amt = post_balance_map.get((mint, user_pubkey), 0)

            decimals = local_info.get("decimals", 0) # local decimals

            diff_raw = post_amt - pre_amt
            diff_float = diff_raw / (10 ** decimals)

            # update holdings balance (local balance)
            holdings[mint]["balance"] = holdings[mint]["balance"] + diff_float
            holdings[mint]["balance"] = post_amt / (10**decimals)

            # update time
            holdings[mint]["last_update_time"] = block_time

            logging.info(
                f"Updated holdings for {mint} by {diff_float:.6f} tokens at blockTime={block_time} "
                f"due to tx={tx_sig}"
            )


def verify_tx_and_get_balance_change(
    client: Client,
    tx_sig: str,
    user_pubkey: str,
    output_mint: str,
    output_decimals: int
) -> Tuple[str, float]:
    """
    Verifies if the transaction went through and calculates the balance change
    of the output token.

    :param client: Client for Solana RPC.
    :param tx_sig: Transaction signature.
    :param user_pubkey: User's public key.
    :param output_mint: The mint address of the output token.
    :param output_decimals: The decimals for the output token.
    :return: Tuple (status, balance change). Status can be 'notFound', 'processed', 'confirmed', or 'finalized'.
    """
    try:
        tx_signature = Signature.from_string(tx_sig)

        # Get transaction response for "Finalized" commitment level
        tx_response = client.get_transaction(tx_signature, max_supported_transaction_version=2)
        if not tx_response.value:
            return "notFound", 0.0, None

        # Extract metadata and perform finalized-level operations
        meta = tx_response.value.transaction.meta
        slot_status = tx_response.value.slot

        # Check finalized status
        tx_status_resp = client.get_signature_statuses([tx_signature], search_transaction_history=True)

        is_finalized = tx_status_resp.value[0].confirmation_status.__str__() == "TransactionConfirmationStatus.Finalized"
        if not is_finalized:
            logging.warning(f"Transaction {tx_sig} is not finalized yet.")
            return "notFinalized", 0.0, None

        tx_err = tx_status_resp.value[0].err

        if tx_err:
            logging.error(f"Transaction {tx_sig} failed with error: {tx_err}")
            return "failed", 0.0, None

        # Process token balance changes if finalized
        pre_balances = meta.pre_token_balances or []
        post_balances = meta.post_token_balances or []

        balance_change = 0.0
        for post in post_balances:
            if str(post.mint) == output_mint and str(post.owner) == user_pubkey:
                post_balance = int(post.ui_token_amount.amount)

                # Match the corresponding pre-balance
                pre_balance = next(
                    (
                        int(pre.ui_token_amount.amount)
                        for pre in pre_balances
                        if str(pre.mint) == output_mint and str(pre.owner) == user_pubkey
                    ),
                    0
                )
                balance_change = (post_balance - pre_balance) / (10**output_decimals)
                break

        # Log balance-change details
        if balance_change > 0:
            return "finalized", balance_change, post_balance
        else:
            logging.warning(f"Transaction {tx_sig} finalized but no balance change detected.")
            return "finalized", 0.0, None

    except Exception as e:
        logging.error(f"Exception occurred while verifying transaction: {e}")
        return "notFound", 0.0, None

def get_jupiter_usd_price(
    output_mint: str,
    output_mint_decimals: int,
    usd: float = 10
    ) -> float:
    """
    Fetches the current price of a token in USD from Jupiter's /price endpoint.
    """
    url = "https://quote-api.jup.ag/v6/price"
        # Step 1: Get a quote
    quote_resp = get_jupiter_quote(
        input_mint='EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v', # USDC
        output_mint=output_mint,
        amount_in=usd*10**6,  # 10 USDC
        max_slippage_bps=300
    )

    if not quote_resp:
        raise ValueError("Failed to get a quote from Jupiter.")

    estimated_amount_out = quote_resp.get("outAmount")

    # convert out amount to float from string
    estimated_amount_out = float(estimated_amount_out) / 10**output_mint_decimals

    price = usd / estimated_amount_out

    return price


def get_jupiter_quote(
    input_mint: str,
    output_mint: str,
    amount_in: int,
    max_slippage_bps: int = 300
) -> dict:
    """
    Fetches a swap quote from Jupiter's /quote endpoint.
    """
    url = "https://quote-api.jup.ag/v6/quote"
    params = {
        "inputMint": input_mint,
        "outputMint": output_mint,
        "amount": str(amount_in),
        "autoSlippage": "true",
        "maxSlippage": str(max_slippage_bps)
    }

    response = requests.get(url, params=params)


    response.raise_for_status()
    return response.json()

def get_jupiter_swap_tx(user_pubkey: str, quote_response: dict) -> str:
    """
    Requests a prepared swap transaction from Jupiter's /swap endpoint.
    """
    url = "https://quote-api.jup.ag/v6/swap"
    payload = {
        "userPublicKey": user_pubkey,
        "quoteResponse": quote_response,
    }

    resp = requests.post(url, json=payload)

    data = resp.json()

    swap_transaction_b64 = data.get("swapTransaction")
    if not swap_transaction_b64:
        raise ValueError("No 'swapTransaction' found in Jupiter's response.")
    return swap_transaction_b64


def sign_and_send_versioned_tx(
    tx_b64: str,
    signer: Keypair,
    client: Client
) -> str:
    """
    Deserializes a base64-encoded VersionedTransaction, signs it using the provided Keypair,
    and submits it to the Solana RPC.

    :param tx_b64:  Base64-encoded transaction string from Jupiter.
    :param signer:  A solders.keypair.Keypair object that will sign the transaction.
    :param client:  A solana.rpc.api.Client instance for sending the transaction.
    :return:        The transaction signature (string) from the network.
    """
    raw_transaction = VersionedTransaction.from_bytes(
        base64.b64decode(tx_b64)
    )

    raw_message_bytes = message.to_bytes_versioned(raw_transaction.message)
    signature = signer.sign_message(raw_message_bytes)

    signed_transaction = VersionedTransaction.populate(
        raw_transaction.message, [signature]
    )

    opts = TxOpts(skip_preflight=False, preflight_commitment=Processed)
    result = client.send_raw_transaction(
        txn=bytes(signed_transaction),
        opts=opts
    )

    resp_dict = json.loads(result.to_json())
    tx_signature = resp_dict.get("result")

    if tx_signature is None:
        raise RuntimeError(f"Transaction not confirmed. Full response: {resp_dict}")

    return tx_signature


def swap_tokens(
    private_key_base58: str,
    rpc_url: str,
    input_mint: str = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
    input_mint_decimals: int = 6,
    output_mint_decimals: int = 6,
    output_mint: str = "BjotV424H4UBvrAiGFGjQGztLxoafxM4HSdCXZR6pump",
    amount_in: float = 0.1,
    slippage_bps: int = 500,
    max_retries: int = 3,
    retry_delay: int = 10,
    bird_client: str = None,
    local_holdings: dict = None,
    holdings_lock: threading.RLock = None,
    wallet_address: str = None,
) -> Tuple[Optional[str], Optional[str], float, Optional[int]]:
    """
    Swaps tokens using Jupiter aggregator and retries up to 3 times if it fails.
    """

    client = Client(rpc_url)
    signer = Keypair.from_base58_string(private_key_base58)

    with holdings_lock:
        if local_holdings and output_mint in local_holdings:
            old_balance = local_holdings[output_mint]["balance"]
        elif bird_client and wallet_address:
            old_balance = bird_client.get_token_balance(wallet_address, output_mint)
        else:
            old_balance = 0.0

    amount_in_raw = int(amount_in * 10**input_mint_decimals)

    tx_sig = None
    final_status = None
    actual_out_amount = 0.0
    block_time = None

    local_swap_time = int(time.time())  # basically attempted swap time

    for attempt in range(1, max_retries + 1):
        try:

            quote_resp = get_jupiter_quote(
                input_mint=input_mint,
                output_mint=output_mint,
                amount_in=amount_in_raw,
                max_slippage_bps=slippage_bps,
            )
            if not quote_resp:
                raise ValueError("No quote from Jupiter")

            estimated_amount_out_str = quote_resp.get("outAmount")
            estimated_amount_out = float(estimated_amount_out_str) / (10**output_mint_decimals)

            # get the swap transaction
            swap_tx_b64 = get_jupiter_swap_tx(
                user_pubkey=str(signer.pubkey()),
                quote_response=quote_resp,
            )

            # sign & send
            tx_sig = sign_and_send_versioned_tx(
                tx_b64=swap_tx_b64,
                signer=signer,
                client=client
            )
            logging.info(f"Swap attempt {attempt} - transaction submitted: {tx_sig}")

            # poll for finalization
            status = None
            out_amount = 0.0
            for i in range(6):
                time.sleep(10)
                status, out_amount, _ = verify_tx_and_get_balance_change(
                    client=client,
                    tx_sig=tx_sig,
                    user_pubkey=str(signer.pubkey()),
                    output_mint=output_mint,
                    output_decimals=output_mint_decimals
                )
                if status == "finalized" and out_amount > 0:
                    logging.info(f"Transaction {tx_sig} finalized with out_amount={out_amount}")
                    final_status = "finalized"
                    actual_out_amount = out_amount
                    block_time = get_tx_block_time(client, tx_sig)
                    break

                if status == "failed":
                    raise ValueError(f"Transaction {tx_sig} failed")

            # if we got final_status, break the entire attempt loop
            if final_status == "finalized":
                break

        except Exception as e:
            logging.warning(f"Swap attempt {attempt} error: {e}")
            if attempt < max_retries:
                logging.info(f"Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                logging.error("Max attempts used. Will do fallback check.")
                break

    if final_status != "finalized":
        if not bird_client or not wallet_address:
            logging.error("No BirdEye client or wallet address. No fallback check.Failing swap.")
            return (None, "notFound", 0.0, None)

        time.sleep(15) # wait for birdeye to update

        current_balance = bird_client.get_token_balance(wallet_address, output_mint)

        if current_balance > old_balance:
            # success ?
            final_status = "finalized"
            actual_out_amount = current_balance - old_balance
            block_time = None

            logging.info(f"BirdEye fallback sees we have {current_balance} now (prev {old_balance}), so success.")
        else:
            logging.warning("BirdEye also shows no new balance. Failing swap.")
            return (tx_sig, "notFound", 0.0, None)


    try:
        with holdings_lock:
            old_bal_local = local_holdings.get(output_mint, {}).get("balance", 0.0)
            new_bal_local = old_bal_local + actual_out_amount
            local_holdings[output_mint] = local_holdings.get(output_mint, {})
            local_holdings[output_mint]["balance"] = new_bal_local

            if block_time:
                local_holdings[output_mint]["last_update_time"] = block_time
            else:
                local_holdings[output_mint]["last_update_time"] = local_swap_time

    except Exception as e:
        logging.error(f"Error updating local holdings: {e}")

    return (tx_sig, final_status, actual_out_amount, block_time)


def get_tx_block_time(
    client: Client,
    tx_sig: str,
) -> Optional[int]:
    """
    Fetches the blockTime (epoch seconds) for the given transaction signature,
    if it's finalized. Returns None if not found or if blockTime is missing.
    """

    try:
        tx_signature = Signature.from_string(tx_sig)

        resp = client.get_transaction(tx_signature, max_supported_transaction_version=2)
        if not resp.value or resp.value.block_time is None:
            return None
        return resp.value.block_time

    except Exception as e:
        logging.error(f"Error getting transaction block time: {e}")
        return None


if __name__ == "__main__":
    client = Client(Config.RPC_URL)

    test_tx = "3wqxyp2DjQh53zzHTTEwvHeGG5CMFfVhZzTJbYmuZ6jFtdk6XnRNJYmNLf52HQeiHJaim4gWH9gdLzsjPjkpFe2P"
    user_pubkey = "J8o1XTmj69Zw6ecvBn1DVzAJDHHUekEpets23N2sA489"
    output_mint = "7u8nZijX8J7Psabzt813RApbv9YEWQTepMfm1zxmpump"
    output_decimals = 6

    block_time = get_tx_block_time(
        client=client,
        tx_sig=test_tx
    )

    logging.info(f"Block time: {block_time}")
