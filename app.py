import warnings
import streamlit as st
from styles import set_custom_css, set_sidebar_style, custom_container
from file_upload import upload_file,load_default_file
from pages import home_page, data_overview_page, modeling_page, map_page, contact_page
import pandas as pd
warnings.filterwarnings('ignore')

# Page settings
st.set_page_config(layout="wide")

# Apply custom styles
set_custom_css()
set_sidebar_style()

# Upload file and load data
upload_file()

# Sidebar page selection
page = st.sidebar.selectbox("Menu", ["Home", "Data Overview", "Modeling", "Map", "Contact"])

# Page routing
if page == "Home":
    home_page()
elif page == "Data Overview":
    data_overview_page()
elif page == "Modeling":
    modeling_page()
elif page == "Map":
    map_page()
elif page == "Contact":
    contact_page()
