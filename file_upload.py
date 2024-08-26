import pandas as pd
import streamlit as st
import requests
import tempfile


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


def download_file_from_google_sheets(file_id):
    """Download a Google Sheets file as an Excel file and return a file-like object."""
    url = f'https://docs.google.com/spreadsheets/d/{file_id}/export?format=xlsx'
    response = requests.get(url)
    response.raise_for_status()  # Ensure we notice bad responses
    return response.content


def load_default_file():
    """Load default file from Google Sheets and return DataFrame."""
    file_id = '19VBAxnZP0cSIeHNNpv1InRzZDBddPM1M'  # Google Sheets dosyanızın ID'si
    try:
        file_content = download_file_from_google_sheets(file_id)

        # Create a temporary file and write the downloaded content to it
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as temp_file:
            temp_file.write(file_content)
            temp_file.flush()
            temp_file_path = temp_file.name

        # Read the file into a DataFrame
        df = pd.read_excel(temp_file_path)
        df = check_date_column(df)
        if df is not None:
            st.session_state['data'] = df
            st.sidebar.info('Default data file loaded from Google Sheets.')
        return df
    except Exception as e:
        st.sidebar.error(f'Error loading default file from Google Sheets: {e}')
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

