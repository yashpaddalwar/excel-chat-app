import os
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_together import ChatTogether
from excel_chat_app.logger_config import logger

def setup_llm(provider: str, api_key: str):
    logger.info(f"Attempting to set up LLM for provider: {provider}")
    """
    Initializes and returns an LLM instance based on the specified provider.

    Args:
        provider (str): The name of the LLM provider (e.g., "Groq", "OpenAI").
        api_key (str): The API key for the selected provider.

    Returns:
        An instance of the LLM class, or None if initialization fails.
    """
    
    os.environ["GROQ_API_KEY"] = ""
    os.environ["OPENAI_API_KEY"] = ""
    os.environ["GOOGLE_API_KEY"] = ""
    os.environ["TOGETHER_API_KEY"] = ""

    try:
        if provider == "Groq":
            os.environ["GROQ_API_KEY"] = api_key
            llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0,
                api_key=api_key
            )
        elif provider == "OpenAI":
            os.environ["OPENAI_API_KEY"] = api_key
            llm = ChatOpenAI(
                model="gpt-4",
                temperature=0,
                api_key=api_key
            )
        elif provider == "Google Gemini":
            os.environ["GOOGLE_API_KEY"] = api_key
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-pro",
                temperature=0,
                google_api_key=api_key
            )
        elif provider == "Together AI":
            os.environ["TOGETHER_API_KEY"] = api_key
            llm = ChatTogether(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                temperature=0,
                api_key=api_key
            )
        else:
            logger.error(f"Unsupported LLM provider: {provider}")
            raise ValueError(f"Unsupported LLM provider: {provider}")
        
        logger.info(f"Successfully set up LLM for provider: {provider}")
        return llm
    except Exception as e:
        logger.error(f"Error setting up LLM for {provider}: {e}", exc_info=True)
        raise ValueError(f"Failed to initialize LLM for {provider}. Please check your API key and provider settings.")
