import streamlit as st
import pandas as pd
import requests

# FastAPI server URL
fast_api_server_url = "http://127.0.0.1:8000"


# Function to get available tables from FastAPI
@st.cache_data
def get_table_names():
    response = requests.get(f"{fast_api_server_url}/api/table_names")
    if response.status_code == 200:
        return response.json()
    st.error("Failed to fetch available tables.")
    return []


# Function to load the selected table using FastAPI
@st.cache_data
def load_table(table_name, start=0, stop=10, features=None):
    params = {
        "table_name": table_name,
        "start": start,
        "stop": stop,
        "features": features
    }
    response = requests.get(f"{fast_api_server_url}/api/table", params=params)
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


# Main Streamlit app
def main():
    st.title("Data Table Viewer")

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
