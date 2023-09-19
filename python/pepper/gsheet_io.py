from typing import List, Optional
import pandas as pd

from gspread_pandas import Spread


def gsheet_to_df(
    spread: Spread,
    sheet_name: str,
    start_row: int = 2,
    header_rows: int = 3,
    clean_header: bool = True
) -> pd.DataFrame:
    """
    Convert a specific sheet of a Google Spreadsheet into a Pandas DataFrame.

    Parameters
    ----------
    spread : Spread
        The Google Spreadsheet object.
    sheet_name : str
        The name of the sheet within the Google Spreadsheet.
    start_row : int, optional
        The row number to start reading data from (default is 2).
    header_rows : int, optional
        The number of header rows to skip (default is 3).
    clean_header : bool, optional
        Whether to clean the header by removing extra levels (default is True).

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the data from the specified sheet.

    Examples
    --------
    >>> gc = Spread('YOUR_SPREADSHEET_ID')
    >>> data = gsheet_to_df(gc, 'Sheet1', start_row=2, header_rows=3)
    >>> data.head()

    Notes
    -----
    This function requires the `gspread-pandas` library to be installed.
    """
    data = spread.sheet_to_df(
        index=0,
        sheet=sheet_name,
        start_row=start_row,
        header_rows=header_rows
    )
    if clean_header:
        data.columns = data.columns.droplevel([header_rows - 1])
    return data


def df_to_gsheet(
    data: pd.DataFrame,
    spread: Spread,
    sheet_name: str,
    as_code: Optional[List[str]] = None,
    as_fr_FR: Optional[List[str]] = None,
    start: str = "A6",
    index: bool = False,
    headers: bool = False
) -> None:
    """
    Export a Pandas DataFrame to a specific sheet in a Google Spreadsheet.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame to export to Google Sheets.
    spread : Spread
        The Google Spreadsheet object to export to.
    sheet_name : str
        The name of the sheet to export the DataFrame to.
    as_code : List[str], optional
        A list of column names to be treated as code (default is None).
    as_fr_FR : List[str], optional
        A list of column names to be converted to the fr_FR locale format (default is None).
    start : str, optional
        The starting cell (e.g., 'A6') for the data export (default is 'A6').
    index : bool, optional
        Whether to include the DataFrame index in the export (default is False).
    headers : bool, optional
        Whether to include column headers in the export (default is False).

    Returns
    -------
    None

    Examples
    --------
    >>> gc = Spread('YOUR_SPREADSHEET_ID')
    >>> data = pd.DataFrame({'Column1': [1, 2, 3], 'Column2': ['A', 'B', 'C']})
    >>> df_to_gsheet(data, gc, 'Sheet1', as_code=['Column2'], as_fr_FR=['Column1'])

    Notes
    -----
    This function requires the `gspread-pandas` library to be installed.
    It allows exporting a DataFrame to Google Sheets with optional formatting for specific columns.
    Columns specified in `as_code` will be escaped and treated as code.
    Columns specified in `as_fr_FR` will be converted to the fr_FR locale format.
    """
    # Convert None to empty strings    
    def null_convert(s):
        return s.apply(
            lambda x: ""
            if x is None or str(x) in {"nan", "[]", "[nan, nan]"}
            else x
        )

    # Escaping of digital texts as text codes
    # The "'" prefix, which forces the string type, is essential to prevent
    # the output from being interpreted numerically by Google Sheets,
    # which is typically interpreted as a date.
    def escape(s):
        return s.apply(lambda x: f"'{x}")

    # Format numbers for French locale
    def fr_convert(s):
        return s.apply(lambda x: str(x).replace(",", ";").replace(".", ","))
    
    # Intersection of columns sets
    def inter(a, b):
        return list(set(a) & set(b))
    
    # Adjust data (formats)
    exported = data.copy()  # Working copy
    exported = exported.apply(null_convert)  # Clear empty cells

    if as_code:
        as_code = inter(as_code, data.columns)
        exported[as_code] = exported[as_code].apply(escape)

    if as_fr_FR:
        as_fr_FR = inter(as_fr_FR, data.columns)
        exported[as_fr_FR] = exported[as_fr_FR].apply(fr_convert)

    # Clear the sheet content before writing new data
    spread.clear_sheet(sheet=sheet_name)

    # Write the DataFrame to the Google Sheet
    spread.df_to_sheet(
        exported,
        sheet=sheet_name,
        index=index,
        headers=headers,
        start=start
    )
