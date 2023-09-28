# _displays.py
from _dashboard_commons import *
from _api_getters import get_client_data


def pivoted_display(
    client_id: int,
    table_name: str | List[str],
    index: str | List[str],
    columns: str | List[str],
    values: str | List[str]
) -> None:
    """Pivoted display of table."""
    log_call_info(this_f_name(), locals().copy())
    st.header(f"`{table_name.upper()}` pivoted data")
    client_data = get_client_data(table_name, client_id)
    client_data.MONTHS_BALANCE = -client_data.MONTHS_BALANCE
    client_data = client_data.pivot(index=index, columns=columns, values=values)
    st.dataframe(client_data)
