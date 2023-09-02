import os


def get_project_dir() -> str:
    """Returns the project's base directory path.

    Raises
    ------
    RuntimeError
        If the `PROJECT_DIR` environment variable is not set.

    Returns
    -------
    str
        The project's base directory path.
    """
    if project_path := os.getenv("PROJECT_DIR"):
        return project_path
    else:
        raise RuntimeError("The `PROJECT_DIR` environment variable is not set.")


def get_modules_dir() -> str:
    """Returns the project's modules directory path.

    Raises
    ------
    RuntimeError
        If the `PROJECT_DIR` environment variable is not set.

    Returns
    -------
    str
        The project's modules directory path.
    """
    return os.path.join(get_project_dir(), "modules")


def get_dataset_dir() -> str:
    """Returns the project's dataset directory path.

    Raises
    ------
    RuntimeError
        If the `PROJECT_DIR` environment variable is not set.

    Returns
    -------
    str
        The project's dataset directory path.
    """
    return os.path.join(get_project_dir(), "dataset")


def get_img_dir() -> str:
    """Returns the project's images directory path.

    Raises
    ------
    RuntimeError
        If the `PROJECT_DIR` environment variable is not set.

    Returns
    -------
    str
        The project's images directory path.
    """
    return os.path.join(get_project_dir(), "img")


def get_tmp_dir() -> str:
    """Returns the project's tmp directory path.

    Raises
    ------
    RuntimeError
        If the `PROJECT_DIR` environment variable is not set.

    Returns
    -------
    str
        The project's tmp directory path.
    """
    return os.path.join(get_project_dir(), "tmp")


def get_dataset_csv_dir() -> str:
    """Returns the project's dataset CSV directory path.

    Raises
    ------
    RuntimeError
        If the `PROJECT_DIR` environment variable is not set.

    Returns
    -------
    str
        The project's dataset CSV directory path.
    """
    return os.path.join(get_dataset_dir(), "csv")


def get_dataset_pqt_dir() -> str:
    """Returns the project's dataset Parquet directory path.

    Raises
    ------
    RuntimeError
        If the `PROJECT_DIR` environment variable is not set.

    Returns
    -------
    str
        The project's dataset Parquet directory path.
    """
    return os.path.join(get_dataset_dir(), "pqt")
