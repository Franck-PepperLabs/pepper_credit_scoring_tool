# from fastapi import APIRouter, Query
# import logging
# logging.basicConfig(level=logging.INFO)
from _router_commons import *
from home_credit.api import get_table_range


router = APIRouter()
logging.info("<get_table> router started")


@router.get("/api/table")
async def get_table(
    table_name: str,
    start: int = Query(
        0,
        description="Start index for the range (inclusive). "
        "Negative values count from the end of the table."
    ),
    stop: int = Query(
        100,
        description="Stop index for the range (exclusive). "
        "Negative values count from the end of the table."
    )
):
    """
    Get a specified range of rows from a table.

    Parameters
    ----------
    table_name : str
        The name of the table to retrieve data from.

    start : int, optional
        Start index for the range (inclusive). Default is 0.
        Negative values count from the end of the table.

    stop : int, optional
        Stop index for the range (exclusive). Default is 100.
        Negative values count from the end of the table.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the selected range of rows from the table.
    """
    log_call_info("get_table", locals().copy())
    #logging.info(f"get_table({table_name}, {start}, {stop})")
    return get_table_range(table_name, start, stop).to_json(orient="split")

