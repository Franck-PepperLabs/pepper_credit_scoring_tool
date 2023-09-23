from _dashboard_commons import *


# Function to get available tables from FastAPI
@st.cache_data
def get_table_names() -> List[str]:
    """Get a list of available table names."""
    log_call_info(this_f_name())
    from home_credit.api import get_table_names as _get_table_names
    return _get_table_names()


# Function to load the selected table using FastAPI
@st.cache_data
def load_table(table_name, start=0, stop=100) -> pd.DataFrame:
    """Get a specified range of rows from a table."""
    log_call_info(this_f_name(), locals().copy())
    from home_credit.api import get_table_range as _load_table
    return _load_table(table_name, start, stop)


# Main Streamlit app
def main():
    log_main_run()

    st.title("Basic Table Viewer (Direct)")

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
