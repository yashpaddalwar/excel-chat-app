Excel Chat Assistant 📊
A powerful Streamlit application that allows you to chat with your Excel data using natural language queries. Upload your Excel files and ask questions about your data - the AI will generate and execute pandas code to provide insights.
Features

Multi-LLM Support: Works with Groq, OpenAI, Google Gemini, and Together AI
Natural Language Queries: Ask questions about your data in plain English
Excel File Upload: Support for multiple Excel files (.xlsx, .xls)
Intelligent Code Generation: Automatically generates pandas code based on your queries
Data Summaries: Auto-generated comprehensive reports for each loaded dataset
Error Handling: Smart retry mechanism with error analysis and correction
Interactive Chat Interface: Streamlit-based chat interface with message history
Code Transparency: View the generated pandas code for each query

Environment Setup (Optional)
Create a .env file in the project root:
envGROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
TOGETHER_API_KEY=your_together_api_key_here
Alternatively, you can enter your API key directly in the Streamlit interface.