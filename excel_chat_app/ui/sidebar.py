import streamlit as st
import tempfile
import os
import pandas as pd
from excel_chat_app.core.llm_setup import setup_llm
from excel_chat_app.core.data_handler import DataManager
from excel_chat_app.core.chat_workflow import ChatWorkflow
from excel_chat_app.logger_config import logger

def setup_llm_configuration():
    logger.info("Setting up LLM configuration in sidebar.")
    st.sidebar.header("⚙️ LLM Configuration")
    st.sidebar.markdown("Select your LLM provider and enter your API key.")

    provider_options = ["Groq", "OpenAI", "Google Gemini", "Together AI"]
    selected_provider = st.sidebar.radio(
        "Choose LLM Provider:",
        provider_options,
        index=provider_options.index(st.session_state.get("llm_provider", "Groq")),
        key="llm_provider_radio"
    )
    
    if selected_provider != st.session_state.get("llm_provider"):
        st.session_state.llm_provider = selected_provider
        st.session_state.api_key_input = ""
        st.session_state.llm_initialized = False

    api_key_placeholder = {
        "Groq": "Enter your Groq API Key",
        "OpenAI": "Enter your OpenAI API Key",
        "Google Gemini": "Enter your Google API Key",
        "Together AI": "Enter your Together AI API Key",
    }
    
    api_key = st.sidebar.text_input(
        api_key_placeholder[selected_provider],
        type="password",
        value=st.session_state.get("api_key_input", ""),
        key="api_key_text_input"
    )
    st.session_state.api_key_input = api_key

    if st.sidebar.button("Initialize LLM", key="initialize_llm_button"):
        if api_key:
            logger.info(f"Initializing LLM with provider: {selected_provider}")
            try:
                llm = setup_llm(selected_provider, api_key)
                st.session_state.data_manager = DataManager()
                st.session_state.chat_workflow = ChatWorkflow(llm, st.session_state.data_manager)
                st.session_state.llm_initialized = True
                logger.info("LLM initialized successfully.")
                st.success(f"LLM initialized with **{selected_provider}**!")
                st.rerun()
            except ValueError as e:
                logger.error(f"Failed to initialize LLM: {e}", exc_info=True)
                st.error(f"Failed to initialize LLM: {e}")
                st.session_state.llm_initialized = False
        else:
            logger.warning("API key not provided for LLM initialization.")
            st.warning("Please enter your API key.")
    
    if st.session_state.get("llm_initialized", False):
        st.sidebar.success(f"LLM ready with {st.session_state.llm_provider}!")
    else:
        st.sidebar.warning("LLM not initialized.")

def setup_file_uploader():
    st.sidebar.markdown("---")
    st.sidebar.header("📤 Upload Excel Files")
    uploaded_files = st.sidebar.file_uploader(
        "Choose Excel files",
        type=["xlsx", "xls"],
        accept_multiple_files=True,
        help="Upload one or more Excel files."
    )

    if uploaded_files:
        if st.sidebar.button("Load Files", key="load_files_button"):
            logger.info(f"Loading {len(uploaded_files)} file(s).")
            if not st.session_state.get("llm_initialized", False):
                logger.warning("Attempted to load files before LLM initialization.")
                st.warning("Please initialize the LLM first.")
                return

            with st.spinner("Loading files..."):
                success_count = 0
                error_messages = []
                
                # Clean up previous temp files and dataframes
                cleanup_temp_files()
                st.session_state.data_manager.dataframes = {}
                st.session_state.dataframe_summaries = {}

                for uploaded_file in uploaded_files:
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            temp_path = tmp_file.name
                            st.session_state.temp_files.append(temp_path)
                        
                        df_name = st.session_state.data_manager.load_excel_file(
                            temp_path, 
                            original_filename=uploaded_file.name
                        )
                        success_count += 1
                        st.session_state.messages.append({"role": "assistant", "content": f"Loaded **'{df_name}'** successfully!"})

                        if st.session_state.chat_workflow:
                            with st.spinner(f"Generating summary for '{df_name}'..."):
                                summary_report = st.session_state.chat_workflow.generate_dataframe_summary_report(df_name)
                                st.session_state.dataframe_summaries[df_name] = summary_report
                    except Exception as e:
                        error_messages.append(f"Error loading **{uploaded_file.name}**: {e}")

                if success_count > 0:
                    st.session_state.files_loaded = True
                    if error_messages:
                        for error in error_messages:
                            st.warning(error)
                    st.success(f"Successfully loaded {len(uploaded_files)} file(s)!")
                    st.session_state.messages.append({"role": "assistant", "content": "Files loaded! How can I help?"})
                else:
                    for error in error_messages:
                        st.error(error)
                    st.rerun()

def display_loaded_dataframes_info():
    if st.session_state.get("files_loaded", False) and st.session_state.get("data_manager"):
        st.sidebar.markdown("---")
        st.sidebar.header("ℹ️ Loaded DataFrames")
        for name, df in st.session_state.data_manager.dataframes.items():
            st.sidebar.write(f"**{name}**: {df.shape[0]} rows, {df.shape[1]} columns")

def setup_about_section():
    st.sidebar.markdown("---")
    st.sidebar.header("About")
    st.sidebar.info(
        "This app uses LLMs and LangGraph to analyze Excel files. "
        "Built by Yash Paddalwar."
    )

def cleanup_temp_files():
    for temp_file in st.session_state.get("temp_files", []):
        try:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        except Exception as e:
            st.error(f"Error cleaning up temp file {temp_file}: {e}")
    st.session_state.temp_files = []

def clear_session():
    logger.info("Clearing session.")
    cleanup_temp_files()
    st.session_state.clear()
    st.rerun()

def format_chat_history(messages):
    chat_str = ""
    for message in messages:
        chat_str += f"**{message['role'].capitalize()}**: "
        if isinstance(message["content"], pd.DataFrame):
            chat_str += "\n" + message["content"].to_string() + "\n"
        elif isinstance(message["content"], pd.Series):
            chat_str += "\n" + message["content"].to_string() + "\n"
        else:
            chat_str += f"{message['content']}\n"
        
        # if message.get("code"):
        #     chat_str += f"\n*Generated Code*:\n```python\n{message['code']}\n```\n"
        
        if message.get("raw_result") is not None:
            chat_str += f"\n*Raw Execution Output*:\n```text\n"
            if isinstance(message["raw_result"], (pd.DataFrame, pd.Series)):
                chat_str += message["raw_result"].to_string()
            else:
                chat_str += str(message["raw_result"])
            chat_str += "\n```\n"
        chat_str += "\n---\n"
    return chat_str

def create_chat_export_button():
    if st.session_state.get("messages"):
        chat_history_str = format_chat_history(st.session_state.messages)
        st.sidebar.download_button(
            label="Export Chat History",
            data=chat_history_str,
            file_name="chat_history.md",
            mime="text/plain",
        )

def setup_sidebar():
    setup_llm_configuration()
    setup_file_uploader()
    display_loaded_dataframes_info()
    setup_about_section()
    st.sidebar.markdown("---")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Clear Session"):
            clear_session()
    with col2:
        create_chat_export_button()
