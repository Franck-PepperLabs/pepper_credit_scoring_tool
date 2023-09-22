from _set_env import setup_python_path
setup_python_path()  # Update the PYTHONPATH and PROJECT_DIR from .env file

import streamlit as st
import matplotlib.pyplot as plt
from home_credit.nb_macros import get_datablock

# Create a title for the app
st.title("First Streamlit Example")

# Load some data
datablock = get_datablock("application", "AMT_REQ")

# Create a plot
st.write("## Data Visualization")
fig, ax = plt.subplots()
ax.hist(datablock.AMT_REQ_CREDIT_BUREAU_YEAR)
st.pyplot(fig)

# Add some interactivity with widgets
option = st.selectbox("Select an option", ["Option 1", "Option 2", "Option 3"])
st.write("You selected:", option)
