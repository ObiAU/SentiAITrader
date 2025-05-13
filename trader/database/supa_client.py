import os, logging, sys
from typing import Any, Dict, List
from supabase import create_client, Client
from datetime import datetime

from trader.config import Config

url: str = Config.SUPABASE_URL
key: str = Config.SUPABASE_KEY
supabase: Client = create_client(url, key)

    
def is_select_query(query: str) -> bool:
    """
    so we only have select queries and not updates etc.

    Tables are:

    trades
    cultism
    sentiment
    logs
    """
    query = query.strip().lower()
    return query.startswith("select")

def is_update_query(query: str) -> bool:

    """
    Updatable tables are:

    trades
    cultism
    sentiment
    logs
    
    """

    return query.startswith("update")

def execute_sql_query(query: str) -> List[Dict[str, Any]]:
    try:
        if not is_select_query(query):
            raise ValueError("Only SELECT queries are allowed.")
        # dynamic fetch methods via query
        response = supabase.rpc("execute_dynamic_query", {"query": query}).execute()
        return response.data
    except Exception as e:
        print(f"Error executing SQL query: {e}")
        return []
    

def insert_row(table_name: str, data: Dict[str, Any]) -> Any:

    response = supabase.table(table_name).insert(data).execute()
    return response.data


def update_row(query: str) -> None:
    try:
        if not is_update_query(query):
            raise ValueError("Only UPDATE queries are allowed.")

        response = supabase.rpc("execute_dynamic_query", {"query": query}).execute()
        print(f"Successfully updated table: {response}")

    except Exception as e:
        print(f"Error updating record: {e}")


def upsert_row_static(table_name: str, updates: Dict[str, Any]) -> None:

    if "position_id" not in updates:
        raise ValueError("The 'position_id' field is required for upsert operations.")

    response = supabase.table(table_name).upsert(updates).execute()

    if response:
        logging.info(f"Successfully upserted row in table '{table_name}': {updates}")
    else:
        logging.warning(f"Upsert did not return an expected response. Likely failed.")
        

def update_row_static_(table_name: str, record_id: Any, updates: Dict[str, Any]) -> None:
    try:
        supabase.table(table_name).update(updates).eq("id", record_id).execute()
        print(f"Updated record {record_id} in table '{table_name}'.")
    except Exception as e:
        print(f"Error updating record {record_id} in table '{table_name}': {e}")


def execute_query_legacy(table_name: str, query: str, filters: Dict[str, Any] = {}) -> List[Dict[str, Any]]:
    try:
        response = supabase.table(table_name).select(query).execute()
        return response.data
    except Exception as e:
        print(f"Error executing query on table '{table_name}': {e}")
        return []
    



if __name__ == "__main__":

    trade_data = {
        "timestamp": "2024-12-26T14:23:50Z",
        "ticker_symbol": "BTC$sample_address",
        "blockchain": "Bitcoin",
        "transaction_signature": "abc123xyz",
        "entry_exit_price": 101_000,
        "amount": 0.05,
        "buy_sell": "buy",
        "type": "market",
        "bot_executed": "sentitrader"
    }
    # new_trade = insert_row("trades", trade_data)
    # print(f"Inserted new trade row: {new_trade}")

    # trades = execute_sql_query("select * from trades")
    # print("All trades:", trades)
    # update_row_static("trades", 1, {"ticker_symbol": "BTC", "token_name": "Bitcoin"})
    res = insert_row("positions", {"status": "open", "token_address": "test_btc_addr_5"})

    print(res)
    # update_row("update trades set token_address = 'BTC$sample_address' where token_address = 'BTC';")

