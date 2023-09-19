from decouple import config
import os
os.environ["PROJECT_DIR"] = config("PROJECT_DIR", default="")
os.environ["PYTHONPATH"] = config("PYTHONPATH", default="")

import streamlit as st

from home_credit.load import get_var_descs

var_descs = get_var_descs()
features = var_descs.Table.str.cat(var_descs.Column, sep=".")
feature = st.selectbox("Select a feature", features)

"You selected: ", feature
