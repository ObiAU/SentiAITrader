from datetime import datetime
from typing import Any, Dict, Optional
from pydantic import BaseModel

class TokenMetadata(BaseModel):
    """
    Pydantic model to store final token information and metadata.
    """
    address: str
    mc: Optional[float] = None
    realMc: Optional[float] = None
    decimals: Optional[int] = None
    trader: Optional[str] = None
    tx_time: Optional[datetime] = None
    tx_hash: Optional[str] = None
    creation_time: Optional[int] = None  # Unix timestamp
    analysis_results: Optional[Dict[str, Any]] = None # since we changing analysis_results
    open_new_position: bool = False
