from typing import List
import logging
import streamlit as st
import requests
from _build_url import build_url
import pandas as pd
# from home_credit.api import (
#     get_table_names via route /api/table_names,
#     get_table_range via route /api/table
# )

logging.basicConfig(level=logging.INFO)  # Initialize logging level

def log_call_info(callable_name: str, kwargs: dict | None = None) -> None:
    kwargs_str = ",".join(
        [f"{k}={v}" for k, v in kwargs.items()]
    ) if kwargs else ""
    logging.info(f"{callable_name}({kwargs_str})")


# FastAPI server URL
server_params = {
    "scheme": "http",
    "hostname": "localhost",
    "port": 8000
}

def get_response(route: str, query_params: dict | None = None) -> requests.Response:
    query_url = build_url(**server_params, path=route, query=query_params)
    return requests.get(query_url)


# Function to get available tables from FastAPI
@st.cache_data
def get_table_names() -> List[str]:
    """Get a list of available table names."""
    log_call_info("load_table")
    response = get_response("/api/table_names")
    if response.status_code == 200:
        return response.json()
    st.error("Failed to fetch available tables.")
    return []


# Function to load the selected table using FastAPI
@st.cache_data
def load_table(table_name, start=0, stop=10, features=None) -> pd.DataFrame:
    # TODO : features --> implémenter ce filtrage côté API
    """Get a specified range of rows from a table."""
    query_params = locals().copy()
    log_call_info("load_table", query_params)
    response = get_response("/api/table", query_params)
    if response.status_code == 200:
        data_json = response.json()
        return pd.read_json(data_json, orient="split")
    st.error(f"Failed to fetch data for table {table_name}.")
    return None


@st.cache_data
def get_column_descriptions(table_name):
    # Load the table of column descriptions (assuming it's a DataFrame)
    descs = load_table("columns_description", start=0, stop=-1)

    # Filter the descriptions for the specified table_name
    descs = descs[descs.Table.str.startswith(table_name)]

    # Return a dictionary where keys are column names and values are descriptions
    return dict(zip(descs.Column, descs.Description))


def show_table(table_name, table_data):
    if table_data is None:
        return

    st.sidebar.header("Select Columns")    

    # Create a dictionary of column descriptions
    col_desc = get_column_descriptions(table_name)

    # Select the columns to display
    selected_columns = [
        col for col in table_data.columns.tolist()
        if st.sidebar.checkbox(col, value=True, help=col_desc.get(col, ""))
    ]

    # Filter the DataFrame to display selected columns
    selected_data = table_data[selected_columns]

    # Display the selected and filtered table
    st.header(f"Preview: `{table_name.upper()}`")
    st.dataframe(selected_data)


# Initialize the session variable 'n_runs'
if "n_runs" not in st.session_state:
    st.session_state.n_runs = 0

# Main Streamlit app
def main():
    st.session_state.n_runs += 1
    logging.info(f"{'-' * 20} main run {st.session_state.n_runs}")

    st.title("Home Credit Table Viewer")

    # Sidebar controls
    st.sidebar.header("Options")
    table_names = get_table_names()
    table_name = st.sidebar.selectbox("Select a table", table_names)
    start = st.sidebar.number_input("Start", value=0)
    stop = st.sidebar.number_input("Stop", value=10)

    # Load and display the selected table
    table_data = load_table(table_name, start, stop)

    show_table(table_name, table_data)


if __name__ == "__main__":
    main()
