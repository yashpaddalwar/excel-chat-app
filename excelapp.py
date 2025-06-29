import streamlit as st
import pandas as pd
import tempfile
import os
from excelthing import ExcelChatSystem
import traceback
from streamlit_modal import Modal

# Page configuration
st.set_page_config(
    page_title="Excel Chat Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'chat_system' not in st.session_state:
    st.session_state.chat_system = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'files_loaded' not in st.session_state:
    st.session_state.files_loaded = False
if 'temp_files' not in st.session_state:
    st.session_state.temp_files = []
if 'dataframe_summaries' not in st.session_state:
    st.session_state.dataframe_summaries = {}
if 'llm_provider' not in st.session_state:
    st.session_state.llm_provider = "Groq" # Default provider
if 'api_key_input' not in st.session_state:
    st.session_state.api_key_input = ""
if 'llm_initialized' not in st.session_state:
    st.session_state.llm_initialized = False

def cleanup_temp_files():
    """Clean up temporary files"""
    for temp_file in st.session_state.temp_files:
        try:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        except Exception as e:
            st.error(f"Error cleaning up temp file {temp_file}: {e}")
    st.session_state.temp_files = []

def initialize_chat_system_with_key(provider: str, api_key: str):
    """Initialize the Excel chat system with the provided API key and provider."""
    
    os.environ["GROQ_API_KEY"] = ""
    os.environ["OPENAI_API_KEY"] = ""
    os.environ["GOOGLE_API_KEY"] = ""
    os.environ["TOGETHER_API_KEY"] = ""

    if provider == "Groq":
        os.environ["GROQ_API_KEY"] = api_key
    elif provider == "OpenAI":
        os.environ["OPENAI_API_KEY"] = api_key
    elif provider == "Google Gemini":
        os.environ["GOOGLE_API_KEY"] = api_key
    elif provider == "Together AI":
        os.environ["TOGETHER_API_KEY"] = api_key

    try:
        # Re-initialize to ensure the new key is picked up
        st.session_state.chat_system = ExcelChatSystem()
        st.session_state.llm_initialized = True
        st.success(f"LLM initialized with **{provider}**!")
        # Rerun to update the main UI which might depend on LLM being ready
        st.rerun() 
        return True
    except ValueError as e: # Catch the specific error from ExcelChatSystem if API key is missing
        st.error(f"Failed to initialize LLM: {str(e)}")
        st.error(f"Please ensure the **{provider}** API key is correct.")
        st.session_state.llm_initialized = False
        return False
    except Exception as e:
        st.error(f"An unexpected error occurred during LLM initialization: {str(e)}")
        st.session_state.llm_initialized = False
        return False

def process_uploaded_files(uploaded_files):
    """Process uploaded Excel files"""
    if not uploaded_files:
        return False
        
    # Ensure chat system is initialized
    if not st.session_state.llm_initialized:
        st.warning("Please configure and initialize your LLM API key first in the sidebar.")
        return False

    try:
        # Clean up previous temp files and dataframes
        cleanup_temp_files()
        if st.session_state.chat_system:
            st.session_state.chat_system.dataframes = {}
            st.session_state.dataframe_summaries = {} # Clear old summaries

        success_count = 0
        error_messages = []

        for uploaded_file in uploaded_files:
            try:
                # Create temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_path = tmp_file.name
                    st.session_state.temp_files.append(temp_path)
                
                # Load the Excel file with original filename
                df_name = st.session_state.chat_system.load_excel_file(
                    temp_path, 
                    original_filename=uploaded_file.name
                )
                success_count += 1
                st.session_state.messages.append({"role": "assistant", "content": f"Loaded **'{df_name}'** successfully!"})

                # Generate and store summary report immediately
                with st.spinner(f"Generating summary for '{df_name}'..."):
                    summary_report = st.session_state.chat_system.generate_dataframe_summary_report(df_name)
                    st.session_state.dataframe_summaries[df_name] = summary_report

            except Exception as e:
                error_messages.append(f"Error loading **{uploaded_file.name}**: {str(e)}")
        
        if success_count > 0:
            st.session_state.files_loaded = True
            if error_messages:
                for error in error_messages:
                    st.warning(error)
            return True
        else:
            st.error("Failed to load any files. Please try again.")
            for error in error_messages:
                st.error(error)
            return False

    except Exception as e:
        st.error(f"An unexpected error occurred during file processing: {str(e)}")
        return False

def display_dataframe_info():
    """Display LLM-generated information about loaded dataframes with download option."""
    if st.session_state.chat_system and st.session_state.chat_system.dataframes:
        st.subheader("📋 Loaded Files Information")
        
        # Initialize modal (not directly used for showing summary after load, but can be for other purposes)
        for name in st.session_state.chat_system.dataframes.keys():
            report_markdown = st.session_state.dataframe_summaries.get(name, "Summary not yet generated.")
            
            with st.expander(f"📊 **{name}**"):
                st.markdown(report_markdown)

                # Download button for the summary report
                st.download_button(
                    label=f"Download '{name}' Summary (Markdown)",
                    data=report_markdown,
                    file_name=f"{name}_summary.md",
                    mime="text/markdown"
                )
    else:
        st.info("No Excel files loaded yet. Please upload files in the sidebar.")

def main():
    st.title("📊 Excel Chat Assistant")
    st.markdown("Upload your Excel files and ask questions about your data using natural language!")

    # --- Sidebar for LLM API Key Configuration ---
    with st.sidebar:
        st.header("⚙️ LLM Configuration")
        st.markdown("Select your LLM provider and enter your API key.")

        provider_options = ["Groq", "OpenAI", "Google Gemini", "Together AI"]
        selected_provider = st.radio(
            "Choose LLM Provider:",
            provider_options,
            index=provider_options.index(st.session_state.llm_provider) if st.session_state.llm_provider in provider_options else 0,
            key="llm_provider_radio"
        )
        # Update session state on change
        if selected_provider != st.session_state.llm_provider:
            st.session_state.llm_provider = selected_provider
            st.session_state.api_key_input = "" # Clear key if provider changes
            st.session_state.llm_initialized = False # Reset initialization status
            

        api_key_placeholder_text = {
            "Groq": "Enter your Groq API Key (starts with sk_groq_...)",
            "OpenAI": "Enter your OpenAI API Key (starts with sk-...)",
            "Google Gemini": "Enter your Google API Key (starts with AIza...)",
            "Together AI": "Enter your Together AI API Key (starts with sk-...)",
        }
        
        api_key = st.text_input(
            api_key_placeholder_text[st.session_state.llm_provider],
            type="password",
            value=st.session_state.api_key_input,
            key="api_key_text_input"
        )
        
        # Store input in session state for persistence across reruns
        st.session_state.api_key_input = api_key

        if st.button("Initialize LLM", key="initialize_llm_button"):
            if api_key:
                initialize_chat_system_with_key(st.session_state.llm_provider, api_key)
            else:
                st.warning("Please enter your API key.")
        
        if st.session_state.llm_initialized:
            st.success(f"LLM ready with {st.session_state.llm_provider}!")
        else:
            st.warning("LLM not initialized. Please enter your API key and click 'Initialize LLM'.")

    st.sidebar.markdown("---")
    # --- Sidebar for File Upload ---
    with st.sidebar:
        st.header("📤 Upload Excel Files")
        uploaded_files = st.file_uploader(
            "Choose Excel files",
            type=["xlsx", "xls"],
            accept_multiple_files=True,
            help="You can upload multiple Excel files. Each file will be treated as a separate dataframe."
        )

        if uploaded_files:
            if st.button("Load Files", key="load_files_button"):
                with st.spinner("Loading files... This may take a moment."):
                    if process_uploaded_files(uploaded_files):
                        st.success(f"Successfully loaded {len(uploaded_files)} file(s)!")
                        st.session_state.messages.append({"role": "assistant", "content": "Files loaded! How can I help you analyze your data?"})
                        st.rerun() # Rerun to update the main content area with file info
        
        # --- Sidebar for Loaded DataFrames Info ---
        if st.session_state.files_loaded and st.session_state.chat_system:
            st.markdown("---")
            st.header("ℹ️ Loaded DataFrames")
        

            for name, df in st.session_state.chat_system.dataframes.items():
                col1, col2 = st.columns([0.7, 0.3])
                with col1:
                    st.write(f"**{name}**: Shape: {df.shape}, Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        # About section in sidebar
        st.markdown("---")
        st.header("About")
        st.info(
            "This app leverages large language models (LLMs) and LangGraph to perform "
            "data analysis on your Excel files using natural language queries. "
            "It generates and executes pandas code to get insights from your data."
        )
        st.markdown("Built by Yash Paddalwar") # Customize this

    # Main content area
    display_dataframe_info()

    # Chat history display
    st.markdown("---")
    st.subheader("💬 Chat with your Data")
    chat_container = st.container()

    with chat_container:
        # This loop iterates through ALL messages in session state to display them
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.chat_message("user").write(message["content"])
            elif message["role"] == "assistant":
                with st.chat_message("assistant"):
                    # Display the main content of the assistant's message
                    if isinstance(message["content"], pd.DataFrame):
                        st.dataframe(message["content"], use_container_width=True)
                    elif isinstance(message["content"], pd.Series):
                        st.write(message["content"].to_string())
                    else:
                        st.write(message["content"])
                    
                    # Display the generated code in an expander
                    if "code" in message and message["code"]:
                        with st.expander("Show Generated Code"):
                            st.code(message["code"], language="python")
                    
                    # Display the raw execution output in an expander, *after* the code expander
                    if "raw_result" in message and message["raw_result"] is not None:
                        with st.expander("Show Raw Execution Output"):
                            # Check if it's already a string, otherwise convert to string for display
                            if isinstance(message["raw_result"], (pd.DataFrame, pd.Series)):
                                st.code(message["raw_result"].to_string(), language="text")
                            else:
                                st.code(str(message["raw_result"]), language="text")
                                

    # Chat input
    if st.session_state.llm_initialized: # Only show chat input if LLM is initialized
        if st.session_state.files_loaded:
            user_query = st.chat_input("Ask a question about your data (Refer the generated report)...")
            if user_query:
                st.session_state.messages.append({"role": "user", "content": user_query})
                with st.chat_message("user"):
                    st.write(user_query)
                
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing your query..."):
                        if st.session_state.chat_system:
                            try:
                                response = st.session_state.chat_system.chat(user_query)
                                
                                # Prepare the message dictionary to be appended to session state
                                message_to_append = {
                                    "role": "assistant",
                                    "code": response.get("code", "") # Include code if present
                                }

                                if response["error"]:
                                    st.error(f"Error: {response['error']}")
                                    message_to_append["content"] = f"An error occurred: {response['error']}"
                                    
                                elif response["needs_clarification"]:
                                    st.warning(response["result"]) # This is the clarification message
                                    message_to_append["content"] = response["result"]

                                else:
                                    # Determine the content to display directly in the chat bubble
                                    if response["result_type"] == "dataframe":
                                        if isinstance(response.get("raw_result"), pd.DataFrame):
                                            st.dataframe(response["raw_result"], use_container_width=True)
                                            message_to_append["content"] = response["raw_result"] # Store DataFrame directly
                                        else:
                                            st.write(response["result"])
                                            message_to_append["content"] = response["result"] # Store string representation
                                    elif response["result_type"] == "series":
                                        if isinstance(response.get("raw_result"), pd.Series):
                                            st.write(response["raw_result"].to_string())
                                            message_to_append["content"] = response["raw_result"] # Store Series directly
                                        else:
                                            st.write(response["result"])
                                            message_to_append["content"] = response["result"] # Store string representation
                                    else:
                                        st.write(response["result"])
                                        message_to_append["content"] = response["result"] # Store string representation

                                    # Always add raw_result to the message_to_append if it exists in the response
                                    if "raw_result" in response:
                                        message_to_append["raw_result"] = response["raw_result"]
                                    
                                # Append the complete message to session state
                                st.session_state.messages.append(message_to_append)
                                st.rerun()
                                
                            except Exception as e:
                                st.error(f"An unexpected error occurred during chat processing: {traceback.format_exc()}")
                                st.session_state.messages.append({"role": "assistant", "content": "Sorry, I encountered an unexpected error. Please try again."})
                        else:
                            st.warning("Please upload and load Excel files first.")
                            st.session_state.messages.append({"role": "assistant", "content": "Please upload and load Excel files first."})
        else:
            st.info("Upload Excel files in the sidebar to start chatting with your data!")
    else:
        st.info("Please configure and initialize your LLM API key in the sidebar to enable chat functionality.")

    # Persistent cleanup on app close/rerun
    st.sidebar.markdown("---")
    if st.sidebar.button("Clear All Data and Chat", on_click=lambda: (
        cleanup_temp_files(),
        st.session_state.clear(),
        st.experimental_rerun() # Use experimental_rerun to clear the page fully
    )):
        pass # Button click handles state clearing and rerun

if __name__ == "__main__":
    main()