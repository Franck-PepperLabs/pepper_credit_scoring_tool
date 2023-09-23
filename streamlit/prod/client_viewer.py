from _dashboard_commons import *
from _api_getters import get_table_names, get_target, get_main_map, get_client_data


def client_selector() -> pd.DataFrame:
    log_call_info(this_f_name())
    
    # Sidebar
    st.sidebar.title("Client Selection Filters")

    # Load the clients list with targets
    client_targets = get_target()
    target_counts = client_targets.value_counts()

    # Load the main map data
    client_infos = get_main_map()

    # Filter by Target Status
    target_values = []
    if st.sidebar.checkbox(f"Non-Defaulter ({target_counts[0]})", value=False):
        target_values.append(0)
    if st.sidebar.checkbox(f"Defaulter ({target_counts[1]})", value=True):
        target_values.append(1)
    if st.sidebar.checkbox(f"Indeterminate ({target_counts[-1]})", value=True):
        target_values.append(-1)
        
    # Apply Target Status filter
    client_infos = client_infos[client_infos.TARGET.isin(target_values)]
    filtered_clients = client_infos.groupby("SK_ID_CURR").count()
    
    # Filter by Minimum Bureau Tracking
    min_bureau_tracking = st.sidebar.number_input(
        "Minimum Bureau Tracking",
        min_value=0, value=5
    )

    # Filter by Minimum Previous Application Tracking
    min_prev_app_tracking = st.sidebar.number_input(
        "Minimum Previous Application Tracking",
        min_value=0, value=5
    )

    # Apply Minimum Tracking filter
    return filtered_clients[
        (filtered_clients.SK_ID_BUREAU >= min_bureau_tracking)
        & (filtered_clients.SK_ID_PREV >= min_prev_app_tracking)
    ]


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


def client_viewer_main():
    log_main_run()

    # Create a title for the app
    st.title("Client Data Viewer")

    # Filter the clients list
    clients = client_selector()

    # Client Selection
    client_id = st.selectbox(
        f"Select a Client ({clients.shape[0]})", clients.index)
    
    # Display the client data from the 7 tables
    for table_name in get_table_names():
        st.header(f"`{table_name.upper()}` data")
        client_data = get_client_data(table_name, client_id)
        st.dataframe(client_data)

    # Bureau balance pivoted display
    pivoted_display(
        client_id, "bureau_balance",
        "SK_ID_BUREAU", "MONTHS_BALANCE", "STATUS"
    )

    # POS CASH balance pivoted display
    pivoted_display(
        client_id, "pos_cash_balance",
        "SK_ID_PREV", "MONTHS_BALANCE", "NAME_CONTRACT_STATUS"
    )

    # Credit Car balance pivoted display
    pivoted_display(
        client_id, "credit_card_balance",
        "SK_ID_PREV", "MONTHS_BALANCE", "NAME_CONTRACT_STATUS"
    )

    # Installments payments pivoted display
    # table_name = "installments_payments"
    # st.header(f"`{table_name.upper()}` pivoted data")
    # client_data = get_client_data(table_name, client_id)
    # client_data.MONTHS_BALANCE = -client_data.MONTHS_BALANCE
    # client_data = client_data.pivot(
    #     index="SK_ID_PREV", columns="MONTHS_BALANCE", values="STATUS")
    # st.dataframe(client_data)


if __name__ == "__main__":
    init_session()
    client_viewer_main()

