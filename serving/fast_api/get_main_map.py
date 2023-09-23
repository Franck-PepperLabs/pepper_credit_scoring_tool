from _router_commons import *
from home_credit.api import get_main_map as _get_main_map


router = APIRouter()
logging.info("<main_map> router started")


@router.get("/api/main_map")
async def get_main_map():
    log_call_info(this_f_name())
    return _get_main_map().to_json(orient="split")
