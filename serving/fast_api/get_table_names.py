from _router_commons import *
from home_credit.api import get_table_names as _get_table_names


router = APIRouter()
logging.info("<get_table_names> router started")


@router.get("/api/table_names", response_model=List[str])
async def get_table_names():
    """Get a list of available table names."""
    log_call_info(this_f_name())
    return _get_table_names()
