"""Module: pepper/env.py

This module provides functions for retrieving essential project directory paths
based on environment variables. These paths are crucial for various operations
within the project. This module ensures that the necessary environment variable,
`PROJECT_DIR`, is set and raises exceptions if not found or if the specified
directories do not exist.

Functions:
- `get_project_dir() -> str`: Returns the project's base directory path.
- `get_python_dir() -> str`: Returns the project's local python modules directory path.
- `get_dataset_dir() -> str`: Returns the project's dataset directory path.
- `get_img_dir() -> str`: Returns the project's images directory path.
- `get_tmp_dir() -> str`: Returns the project's temporary directory path.
- `get_dataset_csv_dir() -> str`: Returns the project's dataset CSV directory path.
- `get_dataset_pqt_dir() -> str`: Returns the project's dataset Parquet directory path.

These functions are essential for obtaining the paths required to access
project-specific directories, ensuring that the project is correctly configured.

Note: For FastAPI and Streamlit scripts, it's crucial to manually include the
`load_env_and_setup_python_path` function from this module in your script.
This function loads environment variables from a .env file and updates the PYTHONPATH as specified.
Due to import limitations, the 'pepper.env' module cannot be imported directly in these scripts.
However, for notebooks executed within the IDE, this process is transparent,
as the IDE loads the .env file automatically at startup.

"""

import os
import sys
from dotenv import load_dotenv


def load_env_and_setup_python_path():
    """
    Update the PYTHONPATH and PROJECT_DIR from .env file.

    This function should be manually copied and included in FastAPI or Streamlit scripts to ensure they have access to the necessary
    environment variables and project paths. Due to import limitations, the 'pepper.env' module cannot be imported directly
    in these scripts, so the function must be copied into the script itself.

    Notes
    -----
    - To use this function, copy and paste it into your FastAPI or Streamlit script.
    - It loads environment variables from a .env file and updates the PYTHONPATH as specified.
    - Make sure to set the PYTHONPATH environment variable in your .env file to include the necessary project directories.

    Example
    -------
    Here's how you can manually include this function in your FastAPI script:

    ```python
    # main.py
    import os
    import sys
    from dotenv import load_dotenv

    def load_env_and_setup_python_path():
        load_dotenv()
        if python_path := os.getenv("PYTHONPATH"):
            python_path_list = python_path.split(";")
            for path in python_path_list:
                sys.path.insert(0, path)

    load_env_and_setup_python_path()

    # Now you can import modules from the 'pepper' and 'home_credit' packages.
    import pepper.some_module
    import home_credit.another_module
    ```

    """
    # Update the PYTHONPATH and PROJECT_DIR from .env file
    load_dotenv()
    if python_path := os.getenv("PYTHONPATH"):
        python_path_list = python_path.split(";")
        for path in python_path_list:
            sys.path.insert(0, path)


def get_project_dir() -> str:
    """
    Return the project's base directory path.

    Returns
    -------
    str
        The project's base directory path.

    Raises
    ------
    RuntimeError
        If the `PROJECT_DIR` environment variable is not set.
    FileNotFoundError:
        If the project directory does not exist.
    """
    if not (project_path := os.getenv("PROJECT_DIR")):
        raise RuntimeError("The `PROJECT_DIR` environment variable is not set.")
    if not os.path.exists(project_path):
        raise FileNotFoundError("The project directory does not exist.")
    return project_path


def get_python_dir() -> str:
    """
    Return the project's local python modules directory path.

    Returns
    -------
    str
        The project's modules directory path.

    Raises
    ------
    RuntimeError
        If the `PROJECT_DIR` environment variable is not set.
    """
    return os.path.join(get_project_dir(), "python")


def get_dataset_dir() -> str:
    """
    Return the project's dataset directory path.

    Returns
    -------
    str
        The project's dataset directory path.

    Raises
    ------
    RuntimeError
        If the `PROJECT_DIR` environment variable is not set.
    """
    return os.path.join(get_project_dir(), "dataset")


def get_img_dir() -> str:
    """
    Return the project's images directory path.

    Returns
    -------
    str
        The project's images directory path.

    Raises
    ------
    RuntimeError
        If the `PROJECT_DIR` environment variable is not set.
    """
    return os.path.join(get_project_dir(), "img")


def get_tmp_dir() -> str:
    """
    Return the project's tmp directory path.

    Returns
    -------
    str
        The project's tmp directory path.

    Raises
    ------
    RuntimeError
        If the `PROJECT_DIR` environment variable is not set.
    """
    return os.path.join(get_project_dir(), "tmp")


def get_dataset_csv_dir() -> str:
    """
    Return the project's dataset CSV directory path.

    Returns
    -------
    str
        The project's dataset CSV directory path.

    Raises
    ------
    RuntimeError
        If the `PROJECT_DIR` environment variable is not set.
    """
    return os.path.join(get_dataset_dir(), "csv")


def get_dataset_pqt_dir() -> str:
    """
    Return the project's dataset Parquet directory path.

    Returns
    -------
    str
        The project's dataset Parquet directory path.

    Raises
    ------
    RuntimeError
        If the `PROJECT_DIR` environment variable is not set.
    """
    return os.path.join(get_dataset_dir(), "pqt")
