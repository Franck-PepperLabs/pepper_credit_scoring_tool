from _router_commons import *
from home_credit.api import get_client_data as _get_client_data


router = APIRouter()
logging.info("<client_data> router started")


@router.get("/api/client_data")
async def get_client_data(table_name: str, client_id: int):
    kwargs = locals().copy()
    log_call_info(this_f_name(), kwargs)
    return _get_client_data(**kwargs).to_json(orient="split")
