# home_credit.customer_data_viewer.py

import os
import sys
from dotenv import load_dotenv
import pandas as pd
import streamlit as st
from home_credit.load import get_target, get_main_map, get_table
from home_credit.merge import currentize
# from home_credit.utils import get_table_names


def setup_python_path():
    # Update the PYTHONPATH and PROJECT_DIR from .env file
    load_dotenv()
    if python_path := os.getenv("PYTHONPATH"):
        python_path_list = python_path.split(";")
        for path in python_path_list:
            sys.path.insert(0, path)


@st.cache_data
def get_client_targets() -> pd.DataFrame:
    # Load the client targets
    return get_target()


@st.cache_data
def get_client_main_map() -> pd.DataFrame:
    # Load the main map data
    return get_main_map()


@st.cache_data
def get_client_data(table_name, client_id) -> pd.DataFrame:
    # Load the main map data
    data = get_table(table_name)
    
    # Check if the table requires currentization, e.g., 'bureau_balance'
    if "SK_ID_CURR" not in data:
        currentize(data)
        data.dropna(inplace=True)
        data.SK_ID_CURR = data.SK_ID_CURR.astype(int)
    
    # Filter and return client-specific data
    return data[data.SK_ID_CURR == client_id]


def get_table_names():
    # Hard-coded list to set the selection and order of sections
    return [
        "application", "bureau", "bureau_balance",
        "previous_application", "pos_cash_balance",
        "credit_card_balance", "installments_payments"
    ]

def client_selector() -> pd.DataFrame:
    # Sidebar
    st.sidebar.title("Client Selection Filters")

    # Load the clients list with targets
    client_targets = get_client_targets()
    target_counts = client_targets.value_counts()

    # Load the main map data
    client_infos = get_client_main_map()

    # Filter by Target Status
    target_values = []
    if st.sidebar.checkbox(f"Non-Defaulter ({target_counts[0]})", value=True):
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
        min_value=0, value=0
    )

    # Filter by Minimum Previous Application Tracking
    min_prev_app_tracking = st.sidebar.number_input(
        "Minimum Previous Application Tracking",
        min_value=0, value=0
    )

    # Apply Minimum Tracking filter
    return filtered_clients[
        (filtered_clients.SK_ID_BUREAU >= min_bureau_tracking)
        & (filtered_clients.SK_ID_PREV >= min_prev_app_tracking)
    ]


def main():
    # Set up Python path
    setup_python_path()

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
    table_name = "bureau_balance"
    st.header(f"`{table_name.upper()}` pivoted data")
    client_data = get_client_data(table_name, client_id)
    client_data.MONTHS_BALANCE = -client_data.MONTHS_BALANCE
    client_data = client_data.pivot(
        index="SK_ID_BUREAU", columns="MONTHS_BALANCE", values="STATUS")
    st.dataframe(client_data)

    # POS CASH balance pivoted display
    table_name = "pos_cash_balance"
    st.header(f"`{table_name.upper()}` pivoted data")
    client_data = get_client_data(table_name, client_id)
    client_data.MONTHS_BALANCE = -client_data.MONTHS_BALANCE
    client_data = client_data.pivot(
        index="SK_ID_PREV", columns="MONTHS_BALANCE", values="NAME_CONTRACT_STATUS")
    st.dataframe(client_data)

    # Credit Car balance pivoted display
    table_name = "credit_card_balance"
    st.header(f"`{table_name.upper()}` pivoted data")
    client_data = get_client_data(table_name, client_id)
    client_data.MONTHS_BALANCE = -client_data.MONTHS_BALANCE
    client_data = client_data.pivot(
        index="SK_ID_PREV", columns="MONTHS_BALANCE", values="NAME_CONTRACT_STATUS")
    st.dataframe(client_data)

    # Installments payments pivoted display
    # table_name = "installments_payments"
    # st.header(f"`{table_name.upper()}` pivoted data")
    # client_data = get_client_data(table_name, client_id)
    # client_data.MONTHS_BALANCE = -client_data.MONTHS_BALANCE
    # client_data = client_data.pivot(
    #     index="SK_ID_PREV", columns="MONTHS_BALANCE", values="STATUS")
    # st.dataframe(client_data)


if __name__ == "__main__":
    main()
