import streamlit as st

def set_page_config():
    st.set_page_config(
        page_title="DataCare Advanced - Medical Data Science Lab",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded",
    )

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)