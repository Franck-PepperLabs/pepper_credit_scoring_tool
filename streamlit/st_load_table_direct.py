import streamlit as st

import pandas as pd
# st.set_option('server.maxMessageSize', 500)  # default is 200 Mb

import requests

# FastAPI server URL
fast_api_server_url = "http://localhost:8000"

# Function to get available tables from FastAPI
def get_table_names():
    # Replace with your FastAPI server URL
    response = requests.get(f"{fast_api_server_url}/api/table_names")
    if response.status_code == 200:
        return response.json()
    st.error("Failed to fetch available tables.")
    return []


# Function to load the selected table using FastAPI
def load_table(table_name, start=0, stop=100):
    response = requests.get(
        f"{fast_api_server_url}/api/table?table_name={table_name}&start={start}&stop={stop}"
    )
    if response.status_code == 200:
        data_json = response.json()
        return pd.read_json(data_json, orient="split")
    st.error(f"Failed to fetch data for table {table_name}.")
    return None


# Main Streamlit app
def main():
    st.title("Data Table Viewer")

    # User input: select the table to load
    table_names = get_table_names()
    selected_table = st.selectbox("Select a table", table_names)

    # Load and display the selected table
    table_data = load_table(selected_table)
    st.write("Table Preview:")
    st.dataframe(table_data)

if __name__ == "__main__":
    main()
