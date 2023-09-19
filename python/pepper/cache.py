from typing import Callable
import re
import threading
import pandas as pd


class Cache:

    _data = {}
    _loaders = {}
    _lock = threading.Lock()

    @staticmethod
    def _is_identifier(identifier: str) -> bool:
        """
        Check if the given string is a valid identifier.

        Parameters
        ----------
        identifier : str
            The string to check

        Returns
        -------
        bool
            True if the string is a valid identifier, False otherwise.

        """
        return re.search(r"^[A-Za-z_][A-Za-z0-9_]*$", identifier) is not None

    @staticmethod
    def clear() -> None:
        """Clear the entire cache."""
        with Cache._lock:
            # critical section, accessing _data and _loader
            Cache._data.clear()
            Cache._loaders.clear()

    @staticmethod
    def exists(table_name: str) -> bool:
        """
        Check if cache exists for a table.

        Parameters
        ----------
        table_name : str
            The name of the table. The table name must be a valid identifier.

        Returns
        -------
        bool
            True if cache exists else False.

        Raises
        ------
        ValueError
            If `table_name` is not a valid identifier.
        """
        if not (
            isinstance(table_name, str)
            and Cache._is_identifier(table_name)
        ):
            raise ValueError(
                f"`table_name` {table_name} is not a valid identifier"
            )
        with Cache._lock:
            # critical section, accessing _data
            return table_name in Cache._data

    @staticmethod
    def get(table_name: str) -> pd.DataFrame:
        """
        Retrieve the cache for a given table.

        Parameters
        ----------
        table_name : str
            The name of the table. The table name must be a valid identifier.

        Returns
        -------
        pd.DataFrame
            The cache of the table if it exists.

        Raises
        ------
        ValueError
            If `table_name` is not a valid identifier
        KeyError
            If `table_name` is not in the cache
        """
        if not (isinstance(table_name, str) and Cache._is_identifier(table_name)):
            raise ValueError(
                f"`table_name` {table_name} is not a valid identifier"
            )
        with Cache._lock:
            # critical section, accessing _data
            table = Cache._data.get(table_name)
            if table is None:
                raise KeyError(f"No cache found for table '{table_name}'")
            return table

    @staticmethod
    def set(
        table_name: str,
        table: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Set the cache for a given table.

        Parameters
        ----------
        table_name : str
            The name of the table. The table name must be a valid identifier.
        table : Union[pd.DataFrame, None]
            The table to store in the cache.

        Returns
        -------
        Union[pd.DataFrame, None]
            The cache of the table if it exists, None otherwise.

        Raises
        ------
        ValueError
            If `table_name` is not a valid identifier
        """
        if not (
            isinstance(table_name, str)
            and Cache._is_identifier(table_name)
        ):
            raise ValueError(
                f"`table_name` {table_name} is not a valid identifier"
            )
        if not (table is None or isinstance(table, pd.DataFrame)):
            raise TypeError(
                f"`table` {table} is not None or a DataFrame"
                f"(type is {type(table)})"
            )
        with Cache._lock:
            # critical section, accessing _data
            Cache._data[table_name] = table
        return table

    @staticmethod
    def init(
        table_name: str,
        loader: Callable[[], pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Initialize the cache for a table. If the cache for the table does
        not exist, it is loaded using the corresponding cache loader.

        Parameters
        ----------
        table_name : str
            The name of the table. The table name must be a valid identifier.

        Returns
        -------
        Union[pd.DataFrame, None]
            The cache of the table if it exists, None otherwise.

        Raises
        ------
        ValueError
            If `table_name` is not a valid identifier.
        """
        if Cache.exists(table_name):
            return Cache.get(table_name)
        if not isinstance(loader, Callable):
            raise TypeError(f"{loader} is not a Callable")
        Cache._loaders[table_name] = loader
        return Cache.set(table_name, loader())

    @staticmethod
    def _reset_cache(table_name: str) -> pd.DataFrame:
        """
        Reset the cache for a table, by loading it again using the
        corresponding cache loader.

        Parameters
        ----------
        table_name : str
            The name of the table. The table name must be a valid identifier.

        Returns
        -------
        pd.DataFrame
            The cache of the table if it exists, None otherwise.

        Raises
        ------
        ValueError
            If `table_name` is not a valid identifier.
        KeyError
            If `table_name` is not in the loaders cache.
        """
        with Cache._lock:
            # critical section, accessing _loaders
            loader = Cache._loaders.get(table_name)
            if loader is None:
                raise KeyError(f"No cache found for loader '{table_name}'")
        return Cache.set(table_name, loader())
