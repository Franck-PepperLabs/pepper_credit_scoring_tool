from fastapi import APIRouter
import logging
logging.basicConfig(level=logging.INFO)

from typing import List
from home_credit.api import get_table_names as _get_table_names


router = APIRouter()
logging.info("get_table_names router started")

@router.get("/api/table_names", response_model=List[str])
async def get_table_names():
    """
    Get a list of available table names.
    """
    logging.info("get_table_names()")
    return _get_table_names()
