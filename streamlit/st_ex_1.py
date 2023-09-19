from decouple import config
import os
os.environ["PROJECT_DIR"] = config("PROJECT_DIR", default="")
os.environ["PYTHONPATH"] = config("PYTHONPATH", default="")
# st.write(f"PROJECT_DIR: {os.environ.get('PROJECT_DIR')}")
# st.write(f"PYTHONPATH: {os.environ.get('PYTHONPATH')}")


import streamlit as st
import matplotlib.pyplot as plt

from home_credit.nb_macros import get_datablock

# Create a title for the app
st.title("Streamlit Example")

# Load some data (e.g., a CSV file)
datablock = get_datablock("application", "AMT_REQ")
#data = pd.read_csv("data.csv")
# datablock.AMT_REQ_CREDIT_BUREAU_YEAR.plot.hist()

# Create a plot
st.write("## Data Visualization")
fig, ax = plt.subplots()  # Créez un objet figure et un axe
ax.hist(datablock.AMT_REQ_CREDIT_BUREAU_YEAR)
st.pyplot(fig)

# Add some interactivity with widgets
option = st.selectbox("Select an option", ["Option 1", "Option 2", "Option 3"])
st.write("You selected:", option)
