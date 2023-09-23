from _dashboard_commons import *
from _api_getters import get_table_names, get_table, get_column_descriptions


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
def table_viewer_main():
    log_main_run()

    st.title("Home Credit Table Viewer")

    # Sidebar controls
    st.sidebar.header("Options")
    table_names = get_table_names()
    table_name = st.sidebar.selectbox("Select a table", table_names)
    start = st.sidebar.number_input("Start", value=0)
    stop = st.sidebar.number_input("Stop", value=10)

    # Load and display the selected table
    table_data = get_table(table_name, start, stop)
    show_table(table_name, table_data)


if __name__ == "__main__":
    init_session()
    table_viewer_main()
