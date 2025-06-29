import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import json
import traceback
import re
from datetime import datetime

# LangChain imports
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_together import ChatTogether
from dotenv import load_dotenv
# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

load_dotenv()

class ExcelChatState(TypedDict):
    messages: Annotated[list, add_messages]
    user_query: str
    generated_code: str
    execution_result: Any
    error_message: str
    dataframe_info: Dict
    retry_count: int
    final_answer: str
    result_type: str 
    raw_result: Any
    validity: str

class ExcelChatSystem:
    def __init__(self):
        self.dataframes = {}
        self.llm = None
        self.graph = None
        self._setup_llm()
        self._build_graph()
    
    def _setup_llm(self):
        """Setup LLM based on available API keys"""
        try:
            if os.getenv("GROQ_API_KEY"):
                self.llm = ChatGroq(
                    model="llama-3.3-70b-versatile",
                    temperature=0,
                    api_key=os.getenv("GROQ_API_KEY")
                )
                print("Using Groq LLM")
            elif os.getenv("OPENAI_API_KEY"):
                self.llm = ChatOpenAI(
                    model="gpt-4",
                    temperature=0,
                    api_key=os.getenv("OPENAI_API_KEY")
                )
                print("Using OpenAI LLM")
            elif os.getenv("GOOGLE_API_KEY"):
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-pro",
                    temperature=0,
                    google_api_key=os.getenv("GOOGLE_API_KEY")
                )
                print("Using Google Gemini LLM")
            elif os.getenv("TOGETHER_API_KEY"):
                self.llm = ChatTogether(
                    model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                    temperature=0,
                    api_key=os.getenv("TOGETHER_API_KEY")
                )
                print("Using Together AI LLM")
            else:
                raise ValueError("No API key found. Please set one of: GROQ_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY, TOGETHER_API_KEY")
        except Exception as e:
            print(f"Error setting up LLM: {e}")
            raise
    
    def load_excel_file(self, file_path: str, original_filename: str = None, sheet_name: Optional[str] = None) -> str:
        """Load Excel file and return dataframe name using original filename"""
        try:
            if sheet_name:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                # Use original filename if provided
                if original_filename:
                    base_name = os.path.splitext(original_filename)[0]
                    df_name = f"{base_name}_{sheet_name}"
                else:
                    df_name = f"{os.path.splitext(os.path.basename(file_path))[0]}_{sheet_name}"
            else:
                df = pd.read_excel(file_path)
                
                if original_filename:
                    df_name = os.path.splitext(original_filename)[0]
                else:
                    df_name = os.path.splitext(os.path.basename(file_path))[0]
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            self.dataframes[df_name] = df
            print(f"Loaded {df_name} with shape: {df.shape}")
            return df_name
        except Exception as e:
            print(f"Error loading Excel file: {e}")
            raise
    
    def _get_dataframe_info(self) -> Dict:
        """Get comprehensive information about loaded dataframes"""
        info = {}
        for name, df in self.dataframes.items():
            sample_data = df.head(3).to_string(index=False)
            
            # Get column info with better type detection
            col_info = []
            for col in df.columns:
                dtype = str(df[col].dtype)
                non_null = df[col].count()
                unique_vals = df[col].nunique()
                
                # Better sample values handling
                sample_vals = ""
                if dtype in ['object', 'string']:
                    if unique_vals <= 10:
                        unique_list = list(df[col].dropna().unique()[:5])
                        sample_vals = f" (examples: {unique_list})"
                    else:
                        sample_list = list(df[col].dropna().head(3))
                        sample_vals = f" (examples: {sample_list})"
                elif dtype in ['int64', 'float64']:
                    if unique_vals <= 10:
                        sample_vals = f" (values: {sorted(df[col].dropna().unique())})"
                    else:
                        min_val = df[col].min()
                        max_val = df[col].max()
                        sample_vals = f" (range: {min_val} to {max_val})"
                
                # Check for mixed types
                mixed_type_check = ""
                if dtype == 'object':
                    numeric_count = pd.to_numeric(df[col], errors='coerce').notna().sum()
                    if numeric_count > 0 and numeric_count < len(df[col].dropna()):
                        mixed_type_check = " [MIXED: contains both numeric and text]"
                
                col_info.append(f"{col} ({dtype}, {non_null} non-null, {unique_vals} unique{sample_vals}{mixed_type_check})")
            
            info[name] = {
                "shape": df.shape,
                "columns": col_info,
                "sample_data": sample_data,
                "memory_usage": df.memory_usage(deep=True).sum(),
                "has_nulls": df.isnull().any().any()
            }
        
        return info
    
    def _create_code_generation_prompt(self) -> ChatPromptTemplate:
        """Create prompt for pandas code generation"""
        system_prompt = """You are an expert pandas code generator. Your task is to generate ONLY executable pandas code based on user queries about Excel data.

CRITICAL RULES:
1. Output ONLY Python pandas code, no explanations, no markdown, no comments
2. Use the exact dataframe variable names provided in the dataframe info
3. Always use proper pandas syntax and methods
4. Handle data types intelligently based on column information provided
5. For columns marked as [MIXED], use proper data cleaning before operations
6. For aggregations, use appropriate pandas methods (groupby, agg, etc.)
7. For filtering, use boolean indexing with proper data type handling
8. For date operations, convert to datetime if needed using pd.to_datetime()
9. For numeric operations on mixed columns, use pd.to_numeric() with errors='coerce'
10. Return results that can be printed or displayed
11. Use .copy() when modifying dataframes to avoid warnings
12. Always handle mixed data types in columns before operations
13. When working with multiple dataframes, choose the most relevant one based on the query context
14. For groupby operations, ensure the grouping column exists and handle any data type issues
15. Always end your code with a statement that returns/prints the result

DATA TYPE HANDLING RULES:
- If a column is marked as [MIXED], clean it first before any numeric operations
- For text columns that might contain numbers, use pd.to_numeric(errors='coerce') when needed
- For date columns, use pd.to_datetime(errors='coerce') when needed
- Always check column existence before using it

DATAFRAME INFORMATION:
{dataframe_info}

AVAILABLE DATAFRAMES: {dataframe_names}

IMPORTANT: Based on the column information above, you can see which columns have mixed data types. Handle them appropriately.

Generate pandas code for the following query. Return ONLY the executable code that ends with displaying the result:"""

        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{query}")
        ])
    
    def generate_code_node(self, state: ExcelChatState) -> ExcelChatState:
        """Generate pandas code based on user query"""
        try:
            df_info = self._get_dataframe_info()
            df_names = list(self.dataframes.keys())
            
            prompt = self._create_code_generation_prompt()
            
            # Format dataframe info for prompt
            info_str = ""
            for name, info in df_info.items():
                info_str += f"\nDataFrame '{name}':\n"
                info_str += f"Shape: {info['shape']}\n"
                info_str += f"Columns: {', '.join(info['columns'])}\n"
                info_str += f"Sample Data:\n{info['sample_data']}\n"
                info_str += "-" * 50 + "\n"
            
            messages = prompt.format_messages(
                dataframe_info=info_str,
                dataframe_names=df_names,
                query=state["user_query"]
            )
            
            response = self.llm.invoke(messages)
            generated_code = response.content.strip()
            
            # Clean the code (remove markdown formatting if present)
            if "```python" in generated_code:
                generated_code = generated_code.split("```python")[1].split("```")[0].strip()
            elif "```" in generated_code:
                generated_code = generated_code.split("```")[1].split("```")[0].strip()
            
            state["generated_code"] = generated_code
            state["dataframe_info"] = df_info
            
            return state
            
        except Exception as e:
            state["error_message"] = f"Code generation error: {str(e)}"
            return state
    
    def execute_code_node(self, state: ExcelChatState) -> ExcelChatState:
        """Execute the generated pandas code safely"""
        try:
            if not state["generated_code"]:
                state["error_message"] = "No code to execute"
                return state
            
            # Create safe execution environment
            safe_globals = {
                'pd': pd,
                'np': np,
                'datetime': datetime,
                **self.dataframes  # Add all loaded dataframes
            }
            
            safe_locals = {}
            
            # Capture print output
            import io
            import sys
            old_stdout = sys.stdout
            sys.stdout = captured_output = io.StringIO()
            
            try:
                # Execute the code
                exec(state["generated_code"], safe_globals, safe_locals)
                
                # Get printed output
                printed_output = captured_output.getvalue()
                
                # Try to find the result
                result = None
                
                if printed_output.strip():
                    result = printed_output.strip()
                elif safe_locals:
                    # Get the last assigned variable
                    result = list(safe_locals.values())[-1]
                else:
                    # Try to evaluate the last line if it's an exprssion
                    lines = state["generated_code"].strip().split('\n')
                    last_line = lines[-1].strip()
                    
                    if not any(keyword in last_line for keyword in ['=', 'import', 'def', 'class', 'if', 'for', 'while', 'print']):
                        try:
                            result = eval(last_line, safe_globals, safe_locals)
                        except:
                            result = "Code executed successfully"
            
            finally:
                sys.stdout = old_stdout
            
            # Determine result type and store raw result
            state["raw_result"] = result
            
            if isinstance(result, pd.DataFrame):
                state["result_type"] = "dataframe"
                state["execution_result"] = result.to_string(index=True, max_rows=None, max_cols=None)
            elif isinstance(result, pd.Series):
                state["result_type"] = "series"
                state["execution_result"] = result.to_string(max_rows=None)
            elif isinstance(result, (int, float, np.number)):
                state["result_type"] = "numeric"
                state["execution_result"] = str(result)
            else:
                state["result_type"] = "text"
                state["execution_result"] = str(result) if result is not None else "Code executed successfully"
            
            state["error_message"] = ""
            
            return state
            
        except Exception as e:
            error_msg = f"Execution error: {str(e)}\n\nCode that failed:\n{state['generated_code']}"
            state["error_message"] = error_msg
            state["execution_result"] = None
            return state
    
    def should_retry_node(self, state: ExcelChatState) -> str:
        """Decide whether to retry code generation, generate final answer, or ask for clarification"""
        if state["error_message"] and state["retry_count"] < 2:
            return "retry"
        elif state["error_message"] and state["retry_count"] >= 2:
            return "clarify"
        elif state["execution_result"] is not None:
            # Only DataFrames go directly to end, others get LLM processing
            if state.get("result_type") == "dataframe":
                return "end"
            else:
                return "generate_answer"
        return "end"
    
    def retry_code_node(self, state: ExcelChatState) -> ExcelChatState:
        """Retry code generation with error feedback"""
        try:
            state["retry_count"] += 1
            
            # Create enhanced error analysis
            error_analysis = self._analyze_error(state["error_message"], state["generated_code"])
            
            # Enhanced prompt with error feedback and column analysis
            retry_prompt = f"""The previous code failed with error: {state['error_message']}

Previous code:
{state['generated_code']}

ERROR ANALYSIS:
{error_analysis}

DATAFRAME INFORMATION (Pay attention to column types and mixed data):
{self._format_dataframe_info()}

SPECIFIC FIXES NEEDED:
1. Check column names exist exactly as shown above
2. For columns marked [MIXED], handle data type conversion properly
3. Use pd.to_numeric(errors='coerce') for numeric operations on mixed columns
4. Use pd.to_datetime(errors='coerce') for date operations
5. Always ensure the result is displayed/printed at the end

Generate ONLY the corrected executable code that addresses the specific error above:"""

            messages = [SystemMessage(content=retry_prompt), HumanMessage(content=state["user_query"])]
            
            response = self.llm.invoke(messages)
            generated_code = response.content.strip()
            
            # Clean the code
            if "```python" in generated_code:
                generated_code = generated_code.split("```python")[1].split("```")[0].strip()
            elif "```" in generated_code:
                generated_code = generated_code.split("```")[1].split("```")[0].strip()
            
            state["generated_code"] = generated_code
            state["error_message"] = ""  # Clear previous error
            
            return state
            
        except Exception as e:
            state["error_message"] = f"Retry error: {str(e)}"
            return state

    def _analyze_error(self, error_message: str, code: str) -> str:
        """Analyze the error and provide specific guidance"""
        analysis = []
        
        if "KeyError" in error_message:
            analysis.append("- Column name not found. Check exact column names from dataframe info.")
        
        if "ValueError" in error_message and "convert" in error_message:
            analysis.append("- Data type conversion error. Use pd.to_numeric(errors='coerce') for mixed data.")
        
        if "TypeError" in error_message:
            analysis.append("- Type mismatch. Ensure proper data type handling before operations.")
        
        if "AttributeError" in error_message:
            analysis.append("- Method or attribute not found. Check pandas syntax and dataframe structure.")
        
        if "groupby" in code.lower() and ("KeyError" in error_message or "ValueError" in error_message):
            analysis.append("- GroupBy error. Ensure grouping column exists and handle mixed data types.")
        
        return "\n".join(analysis) if analysis else "General error - check syntax and data types."

    def clarify_node(self, state: ExcelChatState) -> ExcelChatState:
        """Handle clarification step when retries are exhausted"""
        clarification_msg = f"""I apologize, but I'm having trouble processing your query: "{state['user_query']}"

The issue appears to be related to data structure or column names. Here's what you can try:

1. Check if the column names in your query match exactly with the available columns
2. Be more specific about which dataframe you want to analyze
3. Rephrase your query with more details

Available dataframes and columns:
{self._get_simple_dataframe_summary()}

Please rephrase your question or provide more specific details."""
        
        state["final_answer"] = clarification_msg
        state["error_message"] = ""
        return state
    
    def _get_simple_dataframe_summary(self) -> str:
        """Get a simple summary of dataframes for clarification"""
        summary = ""
        for name, df in self.dataframes.items():
            summary += f"\n{name}: {list(df.columns)}"
        return summary

    def generate_final_answer_node(self, state: ExcelChatState) -> ExcelChatState:
        """Generate a natural language answer for non-DataFrame results"""
        try:
            answer_prompt = f"""Based on the user's query and the analysis result, provide a clear, natural language answer.

User Query: {state['user_query']}

Analysis Result:
{state['execution_result']}

Result Type: {state['result_type']}

Instructions:
1. Provide a direct, conversational answer to the user's question
2. Format the data in a readable way (e.g., if it's a series, explain what each value represents)
3. DO NOT mention code or technical implementation
4. Focus on the insights and findings
5. Be specific and factual based on the results
6. If it's numeric data, provide context and interpretation
7. Make the answer user-friendly and easy to understand

Answer:"""

            response = self.llm.invoke([HumanMessage(content=answer_prompt)])
            state["final_answer"] = response.content.strip()
            return state
            
        except Exception as e:
            # Fallback to raw result if LLM fails
            state["final_answer"] = str(state["execution_result"])
            return state
    
    def _format_dataframe_info(self) -> str:
        """Format dataframe info for prompts"""
        df_info = self._get_dataframe_info()
        info_str = ""
        for name, info in df_info.items():
            info_str += f"\nDataFrame '{name}':\n"
            info_str += f"Shape: {info['shape']}\n"
            info_str += f"Columns: {', '.join(info['columns'])}\n"
            info_str += f"Sample Data:\n{info['sample_data']}\n"
            info_str += "-" * 50 + "\n"
        return info_str

    def check_query_validity(self, state: ExcelChatState) -> ExcelChatState:
        prompt = f"""You are a query classifier for an Excel data analysis system. Your task is to politely answer the queries which are not related to excel.

Like if the query is greeting, gibberish, or unrelated to data analysis, you should output the reply to it in natural language.
If the query is related to data analysis, you should return PROCEED.

Here is the query: {state["user_query"]}
DO NOT provide any other explanation other than the output.
"""
        response = self.llm.invoke([HumanMessage(content=prompt)])
        print(response)
        state["validity"] = response.content.strip()
        state["final_answer"] = response.content.strip()
        return state

    
    def _build_graph(self):
        """Build the LangGraph workflow"""
        workflow = StateGraph(ExcelChatState)
        
        # Add nodes
        workflow.add_node("check_query_validity", self.check_query_validity)
        workflow.add_node("generate_code", self.generate_code_node)
        workflow.add_node("execute_code", self.execute_code_node)
        workflow.add_node("retry_code", self.retry_code_node)
        workflow.add_node("generate_answer", self.generate_final_answer_node)
        workflow.add_node("clarify", self.clarify_node)
        
        # Set entry point to check_query_validity
        workflow.set_entry_point("check_query_validity")

        # Conditional edges from check_query_validity
        workflow.add_conditional_edges(
            "check_query_validity",
            lambda state: "generate_code" if "PROCEED" in state.get("validity") else END,
        )

        # Continue with existing edges
        workflow.add_edge("generate_code", "execute_code")
        workflow.add_conditional_edges(
            "execute_code",
            self.should_retry_node,
            {
                "retry": "retry_code",
                "generate_answer": "generate_answer",
                "clarify": "clarify",
                "end": END,
            },
        )
        workflow.add_edge("retry_code", "execute_code")
        workflow.add_edge("generate_answer", END)
        workflow.add_edge("clarify", END)

        self.graph = workflow.compile()
    
    def chat(self, query: str) -> Dict[str, Any]:
        """Main chat interface"""
        if not self.dataframes:
            return {
                "error": "No Excel files loaded. Please load Excel files first using load_excel_file().",
                "result": None,
                "code": None,
                "result_type": "text",
                "raw_result": None
            }
        
        # Initialize state
        initial_state = ExcelChatState(
            messages=[],
            user_query=query,
            generated_code="",
            execution_result=None,
            error_message="",
            dataframe_info={},
            retry_count=0,
            final_answer="",
            result_type="text",
            raw_result=None
        )
        
        try:
            # Run the graph
            final_state = self.graph.invoke(initial_state)
            
            if final_state["error_message"] and final_state["retry_count"] >= 2:
                return {
                    "error": None,
                    "result": final_state.get("final_answer", "Please rephrase your query or be more specific."),
                    "code": final_state["generated_code"],
                    "needs_clarification": True,
                    "result_type": "text",
                    "raw_result": None
                }
            elif final_state["error_message"]:
                return {
                    "error": final_state["error_message"],
                    "result": None,
                    "code": final_state["generated_code"],
                    "needs_clarification": False,
                    "result_type": "text",
                    "raw_result": None
                }
            else:
                # Return final answer if available, otherwise raw result
                result_to_return = final_state.get("final_answer") or final_state["execution_result"]
                return {
                    "error": None,
                    "result": result_to_return,
                    "code": final_state["generated_code"],
                    "needs_clarification": False,
                    "result_type": final_state.get("result_type", "text"),
                    "raw_result": final_state.get("raw_result")
                }
                
        except Exception as e:
            return {
                "error": f"System error: {str(e)}",
                "result": None,
                "code": None,
                "needs_clarification": False,
                "result_type": "text",
                "raw_result": None
            }
    
    def get_dataframe_summary(self) -> str:
        """Get summary of loaded dataframes"""
        if not self.dataframes:
            return "No dataframes loaded."
        
        summary = "Loaded DataFrames:\n" + "="*50 + "\n"
        for name, df in self.dataframes.items():
            summary += f"\n{name}:\n"
            summary += f"  Shape: {df.shape}\n"
            summary += f"  Columns: {list(df.columns)}\n"
            summary += f"  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n"
            summary += "-" * 30 + "\n"
        
        return summary
    
    def generate_dataframe_summary_report(self, df_name: str) -> str:
        """
        Generates a detailed markdown summary report for a specific dataframe using the LLM.
        The report includes an overview, column descriptions, and insights.
        """
        if df_name not in self.dataframes:
            return f"Error: DataFrame '{df_name}' not found."

        df = self.dataframes[df_name]
        df_info = self._get_dataframe_info()[df_name]

        # Prepare detailed column information for the prompt
        column_details = []
        for col_name in df.columns:
            col_series = df[col_name]
            dtype = str(col_series.dtype)
            non_null_count = col_series.count()
            unique_count = col_series.nunique()
            
            detail = f"- **'{col_name}'**: Type: `{dtype}`, Non-Null: {non_null_count}, Unique Values: {unique_count}"

            # Add min/max for numeric, date ranges for datetime
            if pd.api.types.is_numeric_dtype(col_series):
                if not col_series.empty:
                    detail += f", Range: [{col_series.min():.2f} - {col_series.max():.2f}]"
            elif pd.api.types.is_datetime64_any_dtype(col_series):
                if not col_series.empty:
                    detail += f", Date Range: [{col_series.min().strftime('%Y-%m-%d')} to {col_series.max().strftime('%Y-%m-%d')}]"
            
            # Add top unique values for categorical/object
            if unique_count > 0 and unique_count <= 10 and not pd.api.types.is_numeric_dtype(col_series):
                top_values = col_series.value_counts().nlargest(5).index.tolist()
                detail += f", Top Values: {top_values}"
            elif unique_count > 10 and dtype == 'object':
                 sample_values = list(col_series.dropna().sample(min(5, unique_count)))
                 detail += f", Sample Values: {sample_values}"

            # Check for mixed types in object columns
            if dtype == 'object':
                numeric_elements = pd.to_numeric(col_series, errors='coerce').notna().sum()
                if numeric_elements > 0 and numeric_elements < non_null_count:
                    detail += " **[WARNING: Mixed data types detected (numeric and text)]**"
            
            column_details.append(detail)

        # Get a few sample rows
        sample_rows_md = df.head(5).to_markdown(index=False)

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are an expert Data Analyst AI. Your task is to generate a comprehensive and insightful summary report in **Markdown format** for an Excel dataset.
            
The report should be structured clearly with headings and bullet points. Focus on providing valuable insights and descriptions, not just raw numbers.

Here's the information about the DataFrame:
- **DataFrame Name**: {df_name}
- **Shape**: {df_shape}
- **Total Memory Usage**: {df_memory_mb:.2f} MB
- **Has Missing Values**: {has_nulls_text}

**Column-wise Details**:
{column_details}

**First 5 Rows (Sample Data)**:
{sample_rows}

Based on this information, generate a markdown report that includes:
---
## Data Summary Report: {df_name}

### 1. Overview and Key Characteristics
- Briefly describe the dataset, its size, and whether it contains missing values.
- Mention any immediate observations about its overall structure.

### 2. Column-wise Description and Insights
- For each column, provide a concise description of its purpose or what it represents.
- Based on the data type and sample values, infer and explain potential uses, data quality issues (like mixed types), or interesting characteristics (e.g., if it's a date column, what kind of events might it track?).
- Highlight the significance of each column.

### 3. Potential Trends and Patterns (Inferential)
- Based on column names and sample data, briefly speculate on any possible trends, relationships between columns, or patterns that might exist in the data (e.g., "The 'Date' column suggests time-series analysis for sales trends," or "The combination of 'Customer Code' and 'Invoice Number' could indicate transaction volume per customer.").
- **Do not perform actual calculations here; only describe potential insights.**

### 4. Data Quality Notes
- Summarize any observed data quality issues, such as missing values or mixed data types, and suggest what they might imply for analysis.

Generate the report in pure markdown. Do not include any preambles or conversational text outside the markdown.
"""),
            ("human", "Generate the detailed summary report for the dataframe.")
        ])

        try:
            messages = prompt_template.format_messages(
                df_name=df_name,
                df_shape=df_info['shape'],
                df_memory_mb=df_info['memory_usage'] / 1024**2,
                has_nulls_text="Yes" if df_info['has_nulls'] else "No",
                column_details="\n".join(column_details),
                sample_rows=sample_rows_md
            )
            response = self.llm.invoke(messages)
            return response.content.strip()
        except Exception as e:
            return f"Error generating summary report: {str(e)}"

