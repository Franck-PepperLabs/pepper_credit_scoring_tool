# _user_controls.py
from _dashboard_commons import *
from _api_getters import get_target, get_main_map


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
