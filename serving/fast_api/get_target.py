from _router_commons import *
from home_credit.api import get_target as _get_target


router = APIRouter()
logging.info("<target> router started")


@router.get("/api/target")
async def get_target():
    log_call_info(this_f_name())
    return _get_target().to_json(orient="split")
