from gspread_pandas import Spread


# GSheet utils
def gsheet_to_df(spread, sheetname, start_row=2, header_rows=3, clean_header=True):
    data = spread.sheet_to_df(index=0, sheet=sheetname, start_row=start_row, header_rows=header_rows)
    if clean_header:
        data.columns = data.columns.droplevel([header_rows - 1])
    return data


def data_to_gsheet(data, spread, sheet_name, as_code=None, as_fr_FR=None, start='A6'):
    # local utils
    esc = lambda s: s.apply(lambda x: '\'' + str(x))   # escaping of digital texts as text codes
    clr = lambda s: s.apply(                           # clear empty cells
        lambda x: '' if x is None or str(x) in ['nan', '[]', '[nan, nan]'] else x)  
    to_fr = lambda s: s.apply(lambda x: str(x)         # convert formats to fr_FR locale
        .replace(',', ';').replace('.', ',')) 
    inter = lambda a, b: list(set(a) & set(b))         # intersection
    # one_of_in = lambda a, b: len(inter(a, b)) > 0
    
    # ajustements des données (formats)
    exported = data.copy()                                   # working copy
    exported = exported.apply(clr)                           # clear empty cells
    
    if as_code:
        as_code = inter(as_code, data.columns)
        exported[as_code] = exported[as_code].apply(esc)
    if as_fr_FR:
        as_fr_FR = inter(as_fr_FR, data.columns)
        exported[as_fr_FR] = exported[as_fr_FR].apply(to_fr)
    
    spread.df_to_sheet(exported, sheet=sheet_name, index=False, headers=False, start=start)
    # display(exported.loc[:, 'filling_rate':'mod_freqs'])  # un dernier contrôle visuel
