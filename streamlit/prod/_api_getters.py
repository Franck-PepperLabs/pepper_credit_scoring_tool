from _dashboard_commons import *


# Function to get available tables from FastAPI
@st.cache_data
def get_table_names() -> List[str]:
    """Get a list of available table names."""
    log_call_info(this_f_name())
    return get_response("/api/table_names", response_type=list)


# Function to load the selected table using FastAPI
@st.cache_data
def get_table(table_name, start=0, stop=10, features=None) -> pd.DataFrame:
    # TODO : features --> implémenter ce filtrage côté API
    """Get a specified range of rows from a table."""
    query_params = locals().copy()
    log_call_info(this_f_name(), query_params)
    return get_response("/api/table", query_params)


@st.cache_data
def get_column_descriptions(table_name) -> Dict[str, str]:
    log_call_info(this_f_name(), locals().copy())
    # Load the table of column descriptions (assuming it's a DataFrame)
    descs = get_table("columns_description", start=0, stop=-1)

    # Filter the descriptions for the specified table_name
    descs = descs[descs.Table.str.startswith(table_name)]

    # Return a dictionary where keys are column names and values are descriptions
    return dict(zip(descs.Column, descs.Description))


@st.cache_data
def get_target() -> pd.DataFrame:
    """Load the client targets."""
    log_call_info(this_f_name())
    return get_response("/api/target")


@st.cache_data
def get_main_map() -> pd.DataFrame:
    """Load the main map data."""
    log_call_info(this_f_name())
    return get_response("/api/main_map")


def get_predict(
    sk_curr_id: int,
    proba: bool = False
) -> Union[int, float]:
    """...."""
    query_params = locals().copy()
    log_call_info(this_f_name(), query_params)
    response_type = float if proba else int
    return get_response("/api/predict", query_params, response_type)


@st.cache_data
def get_client_data(table_name: str, client_id: int) -> pd.DataFrame:
    """Load the client data from table."""
    query_params = locals().copy()
    log_call_info(this_f_name(), query_params)
    return get_response("/api/client_data", query_params)
