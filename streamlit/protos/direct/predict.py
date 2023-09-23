from _dashboard_commons import *
# from home_credit.api import predict as _predict
# from home_credit.load import get_target, get_main_map


@st.cache_data
def get_client_main_map() -> pd.DataFrame:  # async
    """Load the main map data."""
    log_call_info("predict")
    from home_credit.load import get_main_map 
    return get_main_map()


# @st.cache_data
# @st.cache_data(experimental_allow_widgets=True)
def client_selector() -> pd.DataFrame:  # async
    log_call_info("client_selector")
    # Sidebar
    st.sidebar.title("Client Selection Filters")

    # Load the clients list with targets
    client_targets = get_client_targets()
    target_counts = client_targets.value_counts()

    # Load the main map data
    client_infos = get_client_main_map()

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
        "Minimum of External Loans tracked by Bureau",
        min_value=0, value=5
    )

    # Filter by Minimum Previous Application Tracking
    min_prev_app_tracking = st.sidebar.number_input(
        "Minimum of Previous Home Credit Loans",
        min_value=0, value=5
    )

    # Apply Minimum Tracking filter
    return filtered_clients[
        (filtered_clients.SK_ID_BUREAU >= min_bureau_tracking)
        & (filtered_clients.SK_ID_PREV >= min_prev_app_tracking)
    ]


@st.cache_data
def get_client_targets() -> pd.DataFrame:  # async
    """Load the client targets."""
    log_call_info("get_client_targets")
    from home_credit.load import get_target
    return get_target()


def predict(
    sk_curr_id: int,
    proba: bool = False
) -> Union[int, float]:  # async
    """...."""
    log_call_info("predict", locals().copy())
    from home_credit.api import predict as _predict
    return _predict(sk_curr_id, proba)


def main():  # async
    st.session_state.n_runs += 1
    logging.info(f"{'-' * 20} main run {st.session_state.n_runs}")

    st.title("Predict Viewer (Direct)")

    # Filter the clients list
    clients = client_selector()  # await

    # Client Selection
    client_id = st.selectbox(
        f"Select a Client ({clients.shape[0]})", clients.index)

    st.header("Prediction")
    y_pred = predict(client_id, True)  # await
    st.write(f"{client_id} prediction is : {y_pred}")


if __name__ == "__main__":
    init_session()
    main()
    # asyncio.run(main())  # if set, caching must be customized