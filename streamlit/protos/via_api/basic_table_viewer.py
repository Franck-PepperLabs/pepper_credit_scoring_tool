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

# Initialize the session variable 'n_runs'
if "n_runs" not in st.session_state:
    st.session_state.n_runs = 0

# FastAPI server URL
server_params = {
    "scheme": "http",
    "hostname": "localhost",
    "port": 8000
}

# Function to get available tables from FastAPI
def get_table_names() -> List[str]:
    """Get a list of available table names."""
    logging.info("get_table_names()")
    query_url = build_url(**server_params, path="/api/table_names")
    response = requests.get(query_url)
    if response.status_code == 200:
        return response.json()
    st.error("Failed to fetch available tables.")
    return []


# Function to load the selected table using FastAPI
def load_table(table_name, start=0, stop=100) -> pd.DataFrame:
    """Get a specified range of rows from a table."""
    logging.info(f"load_table({table_name}, {start}, {stop})")
    query_url = build_url(
        **server_params,
        path="/api/table",
        query={
            "table_name": table_name,
            "start": start,
            "stop": stop
        }
    )
    response = requests.get(query_url)
    if response.status_code == 200:
        data_json = response.json()
        return pd.read_json(data_json, orient="split")
    st.error(f"Failed to fetch data for table {table_name}.")
    return None


# Main Streamlit app
def main():
    st.session_state.n_runs += 1
    logging.info(f"{'-' * 20} main run {st.session_state.n_runs}")

    st.title("Basic Table Viewer (Via Fast API)")

    # User input: select the table to load
    table_names = get_table_names()
    selected_table = st.selectbox("Select a table", table_names)

    # Load and display the selected table
    table_data = load_table(selected_table)
    st.write("Table Preview:")
    st.dataframe(table_data)

if __name__ == "__main__":
    main()
