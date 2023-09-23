from _dashboard_commons import *


@st.cache_data
def get_target() -> pd.DataFrame:  # async
    """Load the client targets."""
    log_call_info(this_f_name())
    return get_response("/api/target")


@st.cache_data
def get_main_map() -> pd.DataFrame:  # async
    """Load the main map data."""
    log_call_info(this_f_name())
    return get_response("/api/main_map")


def get_predict(
    sk_curr_id: int,
    proba: bool = False
) -> Union[int, float]:  # async
    """...."""
    query_params = locals().copy()
    log_call_info(this_f_name(), query_params)
    response_type = float if proba else int
    return get_response("/api/predict", query_params, response_type)


# @st.cache_data
# @st.cache_data(experimental_allow_widgets=True)
def client_selector() -> pd.DataFrame:  # async
    log_call_info("client_selector")
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


def display_default_risk_gauge(default_risk: float, threshold: float) -> None:
    # Display prediction in a colored gauge
    import plotly.graph_objects as go
    rel_diff = default_risk - threshold
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=default_risk,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Credit default risk"},
        gauge={
            "axis": {"range": [0, 100]},
            "threshold": {
                "line": {"color": "white", "width": 7},
                "thickness": 1,
                "value": threshold
            },
            "steps": [
                {"range": [0, default_risk], "color": "orange"},
                {"range": [default_risk, 100], "color": "green"}
            ],
            "bar": {"color": "red" if rel_diff > 0 else "orange"}
        }
    ))

    st.plotly_chart(fig, use_container_width=True)
    

def show_client_credit_default_risk(client_id: int, threshold: float) -> None:
    st.header("Predicted Default Risk")
    
    # Client predicted risk
    default_risk = 100*(1 - get_predict(client_id, True))  # await
    st.write(f"Client **{client_id}** credit default risk is **{default_risk:.1f} %**")

    # Threshold for decision
    # threshold = 33.33

    # ...
    rel_diff = default_risk - threshold
    abs_diff = abs(rel_diff)
    tol = .1
    if abs_diff < tol:
        st.write(f"Which is at the threshold (**{threshold:.1f}** %)")
    else:
        # Set color based on proximity to the threshold
        side = "above" if rel_diff > 0 else "below"
        st.write(
            f"Which is **{side}** "
            f"the threshold (**{threshold:.1f} %**) "
            f"by **{abs_diff:.1f} %**"
        )

    display_default_risk_gauge(default_risk, threshold)


def main():  # async
    log_main_run()

    st.title("Predict Viewer (Direct)")

    # Filter the clients list
    clients = client_selector()  # await

    # Client Selection
    client_id = st.selectbox(
        f"Select a Client ({clients.shape[0]})", clients.index)

    # Threshold for decision
    threshold = 33.33
    show_client_credit_default_risk(client_id, threshold)


if __name__ == "__main__":
    init_session()
    main()
    # asyncio.run(main())  # if set, caching must be customized