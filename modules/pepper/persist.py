from typing import Optional, List
import glob
from pepper.utils import create_if_not_exist


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
    ext = ext if ext else "*"
    if recursive:
        filenames = glob.glob(f"**/*.{ext}", root_dir=root_dir, recursive=True)
    else:
        filenames = glob.glob(f"*.{ext}", root_dir=root_dir)
    filenames = [filename.replace("\\", "/") for filename in filenames]
    return filenames


def all_to_parquet(datadict, dir, engine, compression):
    create_if_not_exist(dir)
    for name, data in datadict.items():
        print(".", end="")
        data.to_parquet(
            dir + name + ".pqt",
            engine=engine,
            compression=compression
        )
