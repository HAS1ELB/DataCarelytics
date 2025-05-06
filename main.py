import streamlit as st
from tabular_data.config import set_page_config, load_css
from tabular_data.data_processing import handle_tabular_data_analysis
from tabular_data.utils import display_home_page
from tabular_data.ml_models import handle_machine_learning
from medical_chatbot.app import chatbot

# Configure page
set_page_config()

# Load CSS
try:
    load_css("style.css")
except:
    # Fallback inline CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3498db;
        margin-bottom: 1rem;
    }
    .category-box {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        border-left: 5px solid #3498db;
    }
    </style>
    """, unsafe_allow_html=True)

# Main header
st.markdown("<h1 class='main-header'>DataCare</h1>", unsafe_allow_html=True)
st.markdown("<p class='app-subtitle'>A Comprehensive Medical Data Science Platform with Advanced Analytics</p>", unsafe_allow_html=True)

# Modern navigation sidebar
try:
    from streamlit_option_menu import option_menu
    with st.sidebar:
        #st.image("https://img.icons8.com/fluency/96/000000/hospital-3.png", width=80)
        selected = option_menu(
            "Navigation",
            ["Home", "Tabular Data Analysis", "Image Processing", "Text Analysis", "Chatbot"],
            icons=['house', 'table', 'image', 'chat-square-text', 'diagram-3'],
            menu_icon="app-indicator",
            default_index=0,
        )
    app_mode = selected
except:
    # Fallback to standard selectbox
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose a category", 
        ["Home", "Tabular Data Analysis", "Image Processing", "Text Analysis", "Chatbot"]
    )

# Route to appropriate page
if app_mode == "Home":
    display_home_page()
elif app_mode == "Tabular Data Analysis":    
    # Section for Data Preprocessing
    st.markdown("<div class='category-box'>", unsafe_allow_html=True)
    handle_tabular_data_analysis()
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Section for Machine Learning (only shown if data is loaded)
    if 'df' in st.session_state:
        df = st.session_state['df']
        if 'target' in df.columns:
            st.markdown("<div class='category-box'>", unsafe_allow_html=True)
            handle_machine_learning()
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning("⚠️ The dataset does not contain a 'target' column. Machine Learning section requires a target variable.")
    else:
        st.info("Please upload or load a dataset in the Data Preprocessing section to enable Machine Learning.")
elif app_mode == "Image Processing":
    st.markdown("<h2 class='sub-header'>Image Processing</h2>", unsafe_allow_html=True)
    st.info("Image Processing module coming soon!")
elif app_mode == "Text Analysis":
    st.markdown("<h2 class='sub-header'>Text Analysis</h2>", unsafe_allow_html=True)
    st.info("Text Analysis module coming soon!")
elif app_mode == "Deep Learning":
    st.markdown("<h2 class='sub-header'>Chatbot</h2>", unsafe_allow_html=True)
    chatbot()