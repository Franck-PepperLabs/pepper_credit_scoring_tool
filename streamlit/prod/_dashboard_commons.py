# _dashboard_commons.py
from typing import List, Dict, Union, Any
import os, sys, inspect
from dotenv import load_dotenv
import logging
import json

# import asyncio

import pandas as pd
import numpy as np

import requests

import streamlit as st


def setup_python_path():
    """Update the PYTHONPATH and PROJECT_DIR from .env file."""
    load_dotenv()
    if python_path := os.getenv("PYTHONPATH"):
        python_path_list = python_path.split(";")
        for path in python_path_list:
            sys.path.insert(0, path)


def this_f_name() -> str:
    """
    Return the name of the calling function.

    Returns
    -------
    str: The name of the calling function as a string.

    Example
    -------
    >>> def my_function():
    ...     return this_f_name()
    >>> my_function()
    'my_function'
    """
    return inspect.stack()[1][3]


def log_call_info(callable_name: str, kwargs: dict | None = None) -> None:
    """
    Log information about a function call.

    This function logs the name of the callable and its keyword arguments, if provided.

    Parameters
    ----------
    callable_name : str
        The name of the callable function.
    kwargs : dict, optional
        A dictionary of keyword arguments and their values. Defaults to None.

    Examples
    --------
    To log information about a function call:
    >>> log_call_info("my_function", {"arg1": 42, "arg2": "example"})
    >>> log_call_info(this_f_name(), locals())
    """
    kwargs_str = ",".join(
        [f"{k}={v}" for k, v in kwargs.items()]
    ) if kwargs else ""
    logging.info(f"{callable_name}({kwargs_str})")


def log_main_run(callable_name: str):
    """
    Log the start of a main run in the Streamlit app.

    This function increments the 'n_runs' session variable by 1 and logs the run number.
    Parameters
    ----------
    callable_name : str
        The name of the callable function.

    Examples
    --------
    To log the start of a main run:
    >>> def main():
    >>>     log_main_run(this_f_name())
    >>>     ...
    """
    st.session_state.n_runs += 1
    logging.info(
        f"{'-' * 20} {callable_name} "
        f"run {st.session_state.n_runs}"
    )


def init_session():
    """
    Initialize the Streamlit session.

    This function performs the following actions:
    1. Updates the PYTHONPATH and PROJECT_DIR based on the values in the .env file.
    2. Sets up the logging level to INFO.
    3. Initializes the session variable 'n_runs' to 0 if it does not exist.

    Examples
    --------
    To initialize the Streamlit session:
    >>> if __name__ == "__main__":
    >>>     init_session()
    >>>     main()
    """
    # Update the PYTHONPATH and PROJECT_DIR from .env file
    setup_python_path()

    # Initialize logging level
    logging.basicConfig(level=logging.INFO)

    # Activate if necessary
    # st.set_option('server.maxMessageSize', 500)  # default is 200 Mb

    # Initialize the session variable 'n_runs'
    if "n_runs" not in st.session_state:
        st.session_state.n_runs = 0


def build_url(
    scheme: str | None = "http",
    hostname: str | None = "localhost",
    port: int | None = None,
    path: str | None = "/",
    query: dict | None = None,
    fragment: str | None = None 
) -> str:
    """
    Build a URL by combining its components.

    Parameters
    ----------
    scheme (str, optional): The URL scheme (e.g., "http", "https").
        Defaults to "http".
    hostname (str, optional): The hostname or domain name.
        Defaults to "localhost".
    port (int, optional): The port number, if applicable.
    path (str, optional): The path to the resource.
        Defaults to "/".
    query (dict, optional): A dictionary of query parameters.
    fragment (str, optional): The URL fragment identifier.

    Returns
    -------
    str: The constructed URL.

    Raises
    ------
    ValueError: If the types of input parameters are incorrect.
    """
    if not isinstance(scheme, str):
        raise ValueError("Scheme must be a string")
    if not isinstance(hostname, str):
        raise ValueError("Hostname must be a string")
    if port is not None and not isinstance(port, int):
        raise ValueError("Port must be an integer")
    if not isinstance(path, str):
        raise ValueError("Path must be a string")

    url = f"{scheme}://{hostname}"
    if port:
        url += f":{port}"
    url += path
    if query:
        url += "?" + "&".join([f"{k}={v}" for k, v in query.items()])
    if fragment:
        url += f"#{fragment}"
    return url


# FastAPI server host
def get_server_host() -> dict:
    """
    Return the configuration for the server host.

    Returns
    -------
    dict
        A dictionary containing the server configuration with the following keys:
        - "scheme": The URL scheme (e.g., "http", "https").
        - "hostname": The hostname or domain name (e.g., "localhost").
        - "port": The port number (e.g., 8000).

    Examples
    --------
    To retrieve the server host configuration:
    >>> host_config = get_server_host()
    """
    return {
        "scheme": "http",
        "hostname": "localhost",
        "port": 8000
    }


def get_response(
    route: str,
    query_params: dict | None = None,
    response_type: type = pd.DataFrame
) -> Union[pd.DataFrame, dict, Any, None]:
    """
    Fetch data from an API endpoint and return it in the specified format.

    Parameters
    ----------
    route : str
        The API endpoint route to fetch data from.
    query_params : dict, optional
        A dictionary of query parameters to include in the request URL.
    response_type : Type[Union[pd.DataFrame, dict]], optional
        The expected type of the response data.
        Defaults to pd.DataFrame, which indicates that the response
        should be deserialized as a DataFrame.

    Returns
    -------
    Union[pd.DataFrame, dict, None]
        The response data in the specified format, or None if the request failed.

    Raises
    ------
    JSONDecodeError

    Examples
    --------
    To fetch data as a DataFrame:
    >>> data = get_response("/api/some_endpoint")

    To fetch data as a dictionary:
    >>> data = get_response("/api/some_endpoint", response_type=dict)
    """
    # Build the full API URL
    query_url = build_url(**get_server_host(), path=route, query=query_params)
    response = requests.get(query_url)

    if response.status_code != 200:
        st.error(f"Failed to fetch data from {query_url}.")
        return None

    try:
        data_json = response.json()
    except json.JSONDecodeError as e:
        st.error(f"Failed to decode JSON response: {str(e)}")

    if response_type == pd.DataFrame:
        return pd.read_json(data_json, orient="split")

    return data_json
