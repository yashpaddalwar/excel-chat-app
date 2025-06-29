import streamlit as st
import pandas as pd
import traceback
from excel_chat_app.logger_config import logger

def display_dataframe_info():
    logger.info("Displaying dataframe info.")
    if st.session_state.get("data_manager") and st.session_state.data_manager.dataframes:
        st.subheader("📋 Loaded Files Information")
        for name in st.session_state.data_manager.dataframes.keys():
            report_markdown = st.session_state.dataframe_summaries.get(name, "Summary not available.")
            with st.expander(f"📊 **{name}**"):
                st.markdown(report_markdown)
                st.download_button(
                    label=f"Download '{name}' Summary (Markdown)",
                    data=report_markdown,
                    file_name=f"{name}_summary.md",
                    mime="text/markdown"
                )
    else:
        st.info("Upload Excel files in the sidebar to see data summaries.")

def display_chat_history():
    logger.info("Displaying chat history.")
    st.markdown("---")
    st.subheader("💬 Chat with your Data")
    chat_container = st.container()

    with chat_container:
        for message in st.session_state.get("messages", []):
            with st.chat_message(message["role"]):
                if isinstance(message["content"], pd.DataFrame):
                    st.dataframe(message["content"], use_container_width=True)
                elif isinstance(message["content"], pd.Series):
                    st.write(message["content"].to_string())
                else:
                    st.write(message["content"])
                
                # if message.get("code"):
                #     with st.expander("Show Generated Code"):
                #         st.code(message["code"], language="python")
                
                if message.get("raw_result") is not None:
                    with st.expander("Show Raw Execution Output"):
                        if isinstance(message["raw_result"], (pd.DataFrame, pd.Series)):
                            st.code(message["raw_result"].to_string(), language="text")
                        else:
                            st.code(str(message["raw_result"]), language="text")

def handle_chat_input():
    if not st.session_state.get("llm_initialized", False):
        st.info("Please configure and initialize your LLM in the sidebar to start chatting.")
        return

    if not st.session_state.get("files_loaded", False):
        st.info("Upload and load your Excel files to begin the analysis.")
        return

    user_query = st.chat_input("Ask a question about your data...")
    if user_query:
        logger.info(f"User query: {user_query}")
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.write(user_query)
        
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    response = st.session_state.chat_workflow.chat(user_query)
                    logger.info(f"Assistant response: {response}")
                    
                    message_to_append = {"role": "assistant", "code": response.get("code", "")}

                    if response.get("error"):
                        logger.error(f"Error in chat workflow: {response['error']}")
                        st.error(f"Error: {response['error']}")
                        message_to_append["content"] = f"An error occurred: {response['error']}"
                    elif response.get("needs_clarification"):
                        logger.warning(f"Clarification needed: {response['result']}")
                        st.warning(response["result"])
                        message_to_append["content"] = response["result"]
                    else:
                        if response["result_type"] == "dataframe" and isinstance(response.get("raw_result"), pd.DataFrame):
                            st.dataframe(response["raw_result"], use_container_width=True)
                            message_to_append["content"] = response["raw_result"]
                        elif response["result_type"] == "series" and isinstance(response.get("raw_result"), pd.Series):
                            st.write(response["raw_result"].to_string())
                            message_to_append["content"] = response["raw_result"]
                        else:
                            st.write(response["result"])
                            message_to_append["content"] = response["result"]
                        
                        if "raw_result" in response:
                            message_to_append["raw_result"] = response["raw_result"]
                    
                    st.session_state.messages.append(message_to_append)
                    st.rerun()
                except Exception as e:
                    logger.error(f"An unexpected error occurred in chat input: {e}", exc_info=True)
                    st.error(f"An unexpected error occurred: {traceback.format_exc()}")
                    st.session_state.messages.append({"role": "assistant", "content": "An unexpected error occurred."})

def setup_main_content():
    logger.info("Setting up main content.")
    st.title("📊 Excel Chat Assistant")
    st.markdown("Upload your Excel files and ask questions about your data using natural language!")
    
    display_dataframe_info()
    display_chat_history()
    handle_chat_input()
    logger.info("Main content setup complete.")
