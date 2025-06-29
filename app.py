# app.py

import streamlit as st
from excel_chat_app.ui.sidebar import setup_sidebar
from excel_chat_app.ui.main_content import setup_main_content
from excel_chat_app.logger_config import logger

def initialize_session_state():
    logger.info("Initializing session state.")
    if 'llm_provider' not in st.session_state:
        st.session_state.llm_provider = "Groq"
    if 'api_key_input' not in st.session_state:
        st.session_state.api_key_input = ""
    if 'llm_initialized' not in st.session_state:
        st.session_state.llm_initialized = False
    if 'files_loaded' not in st.session_state:
        st.session_state.files_loaded = False
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'temp_files' not in st.session_state:
        st.session_state.temp_files = []
    if 'dataframe_summaries' not in st.session_state:
        st.session_state.dataframe_summaries = {}
    if 'data_manager' not in st.session_state:
        st.session_state.data_manager = None
    if 'chat_workflow' not in st.session_state:
        st.session_state.chat_workflow = None

def main():
    """
    Main function to run the Streamlit application.
    """
    st.set_page_config(
        page_title="Excel Chat Assistant",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    logger.info("Application started.")
    initialize_session_state()
    setup_sidebar()
    setup_main_content()
    logger.info("Application setup complete.")

if __name__ == "__main__":
    main()
