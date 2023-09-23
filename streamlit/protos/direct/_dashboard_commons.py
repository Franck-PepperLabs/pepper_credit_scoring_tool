from typing import List, Union

import os, sys
from dotenv import load_dotenv

import logging

# import asyncio

import pandas as pd
import numpy as np

import streamlit as st


def setup_python_path():
    """Update the PYTHONPATH and PROJECT_DIR from .env file."""
    load_dotenv()
    if python_path := os.getenv("PYTHONPATH"):
        python_path_list = python_path.split(";")
        for path in python_path_list:
            sys.path.insert(0, path)


def log_call_info(callable_name: str, kwargs: dict | None = None) -> None:
    kwargs_str = ",".join(
        [f"{k}={v}" for k, v in kwargs.items()]
    ) if kwargs else ""
    logging.info(f"{callable_name}({kwargs_str})")


def init_session():
    # Update the PYTHONPATH and PROJECT_DIR from .env file
    setup_python_path()

    # Initialize logging level
    logging.basicConfig(level=logging.INFO)

    # Activate if necessary
    # st.set_option('server.maxMessageSize', 500)  # default is 200 Mb

    # Initialize the session variable 'n_runs'
    if "n_runs" not in st.session_state:
        st.session_state.n_runs = 0
