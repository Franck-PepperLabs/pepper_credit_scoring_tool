from _dashboard_commons import *


@st.cache_data
def get_target() -> pd.DataFrame:  # async
    """Load the client targets."""
    log_call_info(this_f_name())
    from home_credit.load import get_target as _get_target
    return _get_target()


@st.cache_data
def get_main_map() -> pd.DataFrame:  # async
    """Load the main map data."""
    log_call_info(this_f_name())
    from home_credit.load import get_main_map as _get_main_map
    return _get_main_map()


@st.cache_data
def get_client_data(table_name: str, client_id: int) -> pd.DataFrame:
    """Load the client data from table."""
    log_call_info(this_f_name(), locals().copy())
    # Lazy imports
    from home_credit.load import get_table
    from home_credit.merge import currentize

    # Load the main map data
    data = get_table(table_name)
    
    # Check if the table requires currentization, e.g., 'bureau_balance'
    if "SK_ID_CURR" not in data:
        currentize(data)
        data.dropna(inplace=True)
        data.SK_ID_CURR = data.SK_ID_CURR.astype(int)
    
    # Filter and return client-specific data
    return data[data.SK_ID_CURR == client_id]


def get_table_names() -> List[str]:
    # Hard-coded list to set the selection and order of sections
    return [
        "application", "bureau", "bureau_balance",
        "previous_application", "pos_cash_balance",
        "credit_card_balance", "installments_payments"
    ]


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


def main():
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
    main()

