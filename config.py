import streamlit as st

def setup_page_config():
    # Page configuration
    st.set_page_config(page_title="Analyse Médicale", page_icon="🩺", layout="wide")

    # Custom CSS
    st.markdown("""
        <style>
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .sidebar .sidebar-content {
            background-color: #f0f2f6;
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        </style>
    """, unsafe_allow_html=True)