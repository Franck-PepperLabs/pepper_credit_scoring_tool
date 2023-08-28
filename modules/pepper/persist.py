from typing import Optional, List, Dict
import glob
from pepper.utils import create_if_not_exist
import pandas as pd


def _get_filenames_glob(
    root_dir: str,
    ext: Optional[str] = None,
    recursive: bool = False
) -> List[str]:
    """
    Returns a list of filenames in the specified directory
    using glob pattern matching.

    Parameters
    ----------
    root_dir : str
        The root directory to search for filenames in.
    ext : str, optional
        The extension to filter the filenames by.
        Defaults to None, which returns all files.
    recursive : bool, optional
        Whether or not to search for filenames recursively.
        Defaults to False.

    Returns
    -------
    List[str]
        A list of filenames found in the directory.
    """
    ext = ext or "*"
    if recursive:
        filenames = glob.glob(f"**/*.{ext}", root_dir=root_dir, recursive=True)
    else:
        filenames = glob.glob(f"*.{ext}", root_dir=root_dir)
    filenames = [filename.replace("\\", "/") for filename in filenames]
    return filenames


def all_to_parquet(
    datadict: Dict[str, pd.DataFrame],
    target_dir: str,
    engine: str = "pyarrow",
    compression: str = "gzip"
) -> None:
    """
    Save the dataframes in the dictionary to Parquet files in the specified directory.

    Parameters
    ----------
    datadict : Dict[str, pd.DataFrame]
        A dictionary containing the dataframes to save, where the keys represent the table names.
    target_dir : str
        The directory to save the Parquet files in.
    engine : str, optional
        The engine to use for writing Parquet files. Defaults to 'pyarrow'.
    compression : str, optional
        The compression algorithm to use. Defaults to 'gzip'.

    Returns
    -------
    None
    """
    create_if_not_exist(target_dir)
    for name, data in datadict.items():
        print(".", end="")
        data.to_parquet(
            target_dir + name + ".pqt",
            engine=engine,
            compression=compression
        )
