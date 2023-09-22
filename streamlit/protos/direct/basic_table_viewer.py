from typing import List

from _set_env import setup_python_path
setup_python_path()  # Update the PYTHONPATH and PROJECT_DIR from .env file

import logging
logging.basicConfig(level=logging.INFO)  # Initialize logging level

import pandas as pd

from home_credit.api import (
    get_table_names as _get_table_names,
    get_table_range as _load_table
)

import streamlit as st
# st.set_option('server.maxMessageSize', 500)  # default is 200 Mb

# Initialize the session variable 'n_runs'
if "n_runs" not in st.session_state:
    st.session_state.n_runs = 0


# Function to get available tables from FastAPI
def get_table_names() -> List[str]:
    """Get a list of available table names."""
    logging.info("get_table_names()")
    return _get_table_names()

# Function to load the selected table using FastAPI
def load_table(table_name, start=0, stop=100) -> pd.DataFrame:
    """Get a specified range of rows from a table."""
    logging.info(f"load_table({table_name}, {start}, {stop})")
    return _load_table(table_name, start, stop)



# Main Streamlit app
def main():
    st.session_state.n_runs += 1
    logging.info(f"{'-' * 20} main run {st.session_state.n_runs}")

    st.title("Basic Table Viewer (Direct)")

    # User input: select the table to load
    table_names = get_table_names()
    selected_table = st.selectbox("Select a table", table_names)

    # Load and display the selected table
    table_data = load_table(selected_table)
    st.write("Table Preview:")
    st.dataframe(table_data)

if __name__ == "__main__":
    main()
