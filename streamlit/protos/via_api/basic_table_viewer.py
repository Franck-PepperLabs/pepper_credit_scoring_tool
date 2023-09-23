from _dashboard_commons import *


# Function to get available tables from FastAPI
@st.cache_data
def get_table_names() -> List[str]:
    """Get a list of available table names."""
    log_call_info(this_f_name())
    response = get_response("/api/table_names")
    if response.status_code == 200:
        return response.json()
    st.error("Failed to fetch available tables.")
    return []


# Function to load the selected table using FastAPI
@st.cache_data
def load_table(table_name, start=0, stop=100) -> pd.DataFrame:
    """Get a specified range of rows from a table."""
    query_params = locals().copy()
    log_call_info(this_f_name(), query_params)
    response = get_response("/api/table", query_params)
    if response.status_code == 200:
        data_json = response.json()
        return pd.read_json(data_json, orient="split")
    st.error(f"Failed to fetch data for table {table_name}.")
    return None


# Main Streamlit app
def main():
    log_main_run()

    st.title("Basic Table Viewer (Via Fast API)")

    # User input: select the table to load
    table_names = get_table_names()
    selected_table = st.selectbox("Select a table", table_names)

    # Load and display the selected table
    table_data = load_table(selected_table)
    st.write("Table Preview:")
    st.dataframe(table_data)

if __name__ == "__main__":
    init_session()
    main()
