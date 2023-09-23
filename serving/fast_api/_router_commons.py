from typing import List, Union

import os, sys, inspect
from dotenv import load_dotenv

import logging

import pandas as pd
import numpy as np

from fastapi import FastAPI, APIRouter, Query


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
    kwargs_str = ",".join(
        [f"{k}={v}" for k, v in kwargs.items()]
    ) if kwargs else ""
    logging.info(f"{callable_name}({kwargs_str})")


# Update the PYTHONPATH and PROJECT_DIR from .env file
setup_python_path()

# Initialize logging level
logging.basicConfig(level=logging.INFO)

