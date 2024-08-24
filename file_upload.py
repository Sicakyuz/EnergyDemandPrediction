import pandas as pd
import streamlit as st

def check_date_column(df):
    """Check and convert 'TARIH' column to datetime format."""
    if 'TARIH' not in df.columns:
        st.error("'TARIH' column not found in the dataset.")
        return None
    else:
        try:
            df['TARIH'] = pd.to_datetime(df['TARIH'], format='%Y-%m-%d', errors='coerce')
            if df['TARIH'].isnull().any():
                st.error("Some date values in the 'TARIH' column could not be converted to datetime format.")
                return None
            return df
        except Exception as e:
            st.error(f"Error converting 'TARIH' column to datetime format: {e}")
            return None

def upload_file():
    """Handle file upload and return DataFrame."""
    uploaded_file = st.sidebar.file_uploader("Upload Data File", type=['csv', 'xlsx'])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            df = check_date_column(df)
            if df is not None:
                st.session_state['data'] = df
                st.sidebar.success('File uploaded successfully.')
            return df
        except Exception as e:
            st.sidebar.error(f'File upload error: {e}')
    else:
        return load_default_file()

def load_default_file():
    """Load default file and return DataFrame."""
    default_file_path = "/Users/asmir/Desktop/MyProjects/EnergyDemandPrediction/Book2.xlsx"
    try:
        df = pd.read_excel(default_file_path)
        df = check_date_column(df)
        if df is not None:
            st.session_state['data'] = df
            st.sidebar.info('Default data file loaded.')
        return df
    except Exception as e:
        st.sidebar.error(f'Error loading default file: {e}')
        return None
