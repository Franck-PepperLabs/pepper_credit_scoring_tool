from _set_env import setup_python_path
setup_python_path()  # Update the PYTHONPATH and PROJECT_DIR from .env file

import streamlit as st

from home_credit.load import get_var_descs

var_descs = get_var_descs()
features = var_descs.Table.str.cat(var_descs.Column, sep=".")
feature = st.selectbox("Select a feature", features)

"You selected: ", feature
