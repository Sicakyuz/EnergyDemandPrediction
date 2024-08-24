import streamlit as st

def set_custom_css():
    """Set custom CSS for the app."""
    custom_css = """
    <style>
    .css-1siy2j7 {
        font-size: 1.5rem !important;
        font-weight: bold !important;
    }
    h1, h2, h3, .stDataFrame, .stMarkdown {
        font-size: 1.1rem !important;
    }
    .stProgress > div > div > div > div {
        height: 20px !important;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

def set_sidebar_style():
    """Set custom style for the sidebar."""
    st.markdown(
        '''
        <style>
        .stSidebar > div:first-child {
            background-color: #f5f5dc;
        }
        </style>
        ''',
        unsafe_allow_html=True
    )

def custom_container(content):
    """Create a custom container for content."""
    st.markdown(
        f"""
        <div style="background-color: #f0f0f0; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
            {content}
        </div>
        """,
        unsafe_allow_html=True
    )
