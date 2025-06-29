import pandas as pd
import numpy as np
from datetime import datetime
import io
import sys
from typing import Dict, Any, TypedDict
from typing_extensions import Annotated

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from excel_chat_app.config import (
    CODE_GENERATION_PROMPT,
    RETRY_PROMPT,
    FINAL_ANSWER_PROMPT,
    QUERY_VALIDITY_PROMPT,
    SUMMARY_REPORT_PROMPT,
)
from excel_chat_app.core.data_handler import DataManager
from excel_chat_app.logger_config import logger

class ExcelChatState(TypedDict):
    """"State for the Excel chat workflow."""
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

class ChatWorkflow:
    def __init__(self, llm, data_manager: DataManager):
        self.llm = llm
        self.data_manager = data_manager
        self.graph = self._build_graph()
        logger.info("ChatWorkflow initialized.")

    def _build_graph(self):
        """Builds the state graph for the chat workflow."""
        workflow = StateGraph(ExcelChatState)
        workflow.add_node("check_query_validity", self.check_query_validity_node)
        workflow.add_node("generate_code", self.generate_code_node)
        workflow.add_node("execute_code", self.execute_code_node)
        workflow.add_node("retry_code", self.retry_code_node)
        workflow.add_node("generate_answer", self.generate_final_answer_node)
        workflow.add_node("clarify", self.clarify_node)

        workflow.set_entry_point("check_query_validity")

        workflow.add_conditional_edges(
            "check_query_validity",
            lambda state: "generate_code" if "PROCEED" in state.get("validity", "") else END
        )
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

        return workflow.compile()

    def check_query_validity_node(self, state: ExcelChatState) -> ExcelChatState:
        """Checks if the user query is valid for processing."""
        logger.info("Checking query validity.")
        prompt = ChatPromptTemplate.from_messages([("human", QUERY_VALIDITY_PROMPT)]).format_messages(user_query=state["user_query"])
        response = self.llm.invoke(prompt)
        validity = response.content.strip()
        state["validity"] = validity
        if "PROCEED" not in validity:
            logger.warning(f"Query failed validity check: {validity}")
            state["final_answer"] = validity
        else:
            logger.info("Query is valid.")
        return state

    def generate_code_node(self, state: ExcelChatState) -> ExcelChatState:
        """Generates executable code based on the user query and available dataframes."""
        logger.info("Generating code.")
        try:
            df_info = self.data_manager.get_dataframe_info()
            df_names = list(self.data_manager.dataframes.keys())
            
            info_str = self._format_dataframe_info(df_info)
            
            prompt = ChatPromptTemplate.from_messages([("system", CODE_GENERATION_PROMPT), ("human", "{query}")]).format_messages(
                dataframe_info=info_str,
                dataframe_names=df_names,
                query=state["user_query"]
            )
            
            response = self.llm.invoke(prompt)
            generated_code = response.content.strip()
            
            if "```python" in generated_code:
                generated_code = generated_code.split("```python")[1].split("```")[0].strip()
            elif "```" in generated_code:
                generated_code = generated_code.split("```")[1].split("```")[0].strip()
            
            logger.info(f"Generated code:\n{generated_code}")
            state["generated_code"] = generated_code
            state["dataframe_info"] = df_info
            return state
        except Exception as e:
            logger.error(f"Code generation error: {e}", exc_info=True)
            state["error_message"] = f"Code generation error: {str(e)}"
            return state

    def execute_code_node(self, state: ExcelChatState) -> ExcelChatState:
        """Executes the generated code and captures the output."""
        logger.info("Executing code.")
        try:
            if not state["generated_code"]:
                logger.warning("No code to execute.")
                state["error_message"] = "No code to execute"
                return state
            
            safe_globals = {
                'pd': pd,
                'np': np,
                'datetime': datetime,
                **self.data_manager.dataframes
            }
            safe_locals = {}
            
            old_stdout = sys.stdout
            sys.stdout = captured_output = io.StringIO()
            
            try:
                exec(state["generated_code"], safe_globals, safe_locals)
                printed_output = captured_output.getvalue()
                
                result = None
                if printed_output.strip():
                    result = printed_output.strip()
                elif safe_locals:
                    result = list(safe_locals.values())[-1]
                else:
                    lines = state["generated_code"].strip().split('\n')
                    last_line = lines[-1].strip()
                    if not any(keyword in last_line for keyword in ['=', 'import', 'def', 'class', 'if', 'for', 'while', 'print']):
                        try:
                            result = eval(last_line, safe_globals, safe_locals)
                        except:
                            result = "Code executed successfully"
            finally:
                sys.stdout = old_stdout
            
            logger.info(f"Code execution successful. Result type: {type(result)}")
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
            logger.error(error_msg, exc_info=True)
            state["error_message"] = error_msg
            state["execution_result"] = None
            return state

    def should_retry_node(self, state: ExcelChatState) -> str:
        """Determines the next action based on the execution result and error message."""
        if state["error_message"] and state.get("retry_count", 0) < 2:
            return "retry"
        elif state["error_message"]:
            return "clarify"
        elif state.get("result_type") == "dataframe":
            return "end"
        else:
            return "generate_answer"

    def retry_code_node(self, state: ExcelChatState) -> ExcelChatState:
        state["retry_count"] = state.get("retry_count", 0) + 1
        error_analysis = self._analyze_error(state["error_message"], state["generated_code"])
        
        info_str = self._format_dataframe_info(self.data_manager.get_dataframe_info())
        
        prompt = ChatPromptTemplate.from_messages([("system", RETRY_PROMPT), ("human", "{query}")]).format_messages(
            error_message=state["error_message"],
            generated_code=state["generated_code"],
            error_analysis=error_analysis,
            dataframe_info=info_str,
            query=state["user_query"]
        )
        
        response = self.llm.invoke(prompt)
        generated_code = response.content.strip()
        
        if "```python" in generated_code:
            generated_code = generated_code.split("```python")[1].split("```")[0].strip()
        elif "```" in generated_code:
            generated_code = generated_code.split("```")[1].split("```")[0].strip()
        
        state["generated_code"] = generated_code
        state["error_message"] = ""
        return state

    def _analyze_error(self, error_message: str, code: str) -> str:
        """Analyzes the error message and provides specific suggestions for fixing the code."""
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
        """Handles cases where the query needs clarification due to data structure issues."""
        clarification_msg = f"""I apologize, but I'm having trouble processing your query: "{state['user_query']}"
The issue appears to be related to data structure or column names. Here's what you can try:
1. Check if the column names in your query match exactly with the available columns
2. Be more specific about which dataframe you want to analyze
3. Rephrase your query with more details
Available dataframes and columns:
{self.data_manager.get_simple_dataframe_summary()}
Please rephrase your question or provide more specific details."""
        state["final_answer"] = clarification_msg
        state["error_message"] = ""
        return state

    def generate_final_answer_node(self, state: ExcelChatState) -> ExcelChatState:
        """Generates the final answer based on the execution result and user query."""
        try:
            prompt = ChatPromptTemplate.from_messages([("human", FINAL_ANSWER_PROMPT)]).format_messages(
                user_query=state["user_query"],
                execution_result=state["execution_result"],
                result_type=state["result_type"]
            )
            response = self.llm.invoke(prompt)
            state["final_answer"] = response.content.strip()
            return state
        except Exception as e:
            state["final_answer"] = str(state["execution_result"])
            return state

    def _format_dataframe_info(self, df_info: Dict) -> str:
        info_str = ""
        for name, info in df_info.items():
            info_str += f"\nDataFrame '{name}':\n"
            info_str += f"Shape: {info['shape']}\n"
            info_str += f"Columns: {', '.join(info['columns'])}\n"
            info_str += f"Sample Data:\n{info['sample_data']}\n"
            info_str += "-" * 50 + "\n"
        return info_str

    def generate_dataframe_summary_report(self, df_name: str) -> str:
        if df_name not in self.data_manager.dataframes:
            return f"Error: DataFrame '{df_name}' not found."

        df = self.data_manager.dataframes[df_name]
        df_info = self.data_manager.get_dataframe_info()[df_name]

        column_details = []
        for col_name in df.columns:
            col_series = df[col_name]
            dtype = str(col_series.dtype)
            non_null_count = col_series.count()
            unique_count = col_series.nunique()
            
            detail = f"- **'{col_name}'**: Type: `{dtype}`, Non-Null: {non_null_count}, Unique Values: {unique_count}"
            if pd.api.types.is_numeric_dtype(col_series) and not col_series.empty:
                detail += f", Range: [{col_series.min():.2f} - {col_series.max():.2f}]"
            elif pd.api.types.is_datetime64_any_dtype(col_series) and not col_series.empty:
                detail += f", Date Range: [{col_series.min().strftime('%Y-%m-%d')} to {col_series.max().strftime('%Y-%m-%d')}]"
            
            if unique_count > 0 and unique_count <= 10 and not pd.api.types.is_numeric_dtype(col_series):
                top_values = col_series.value_counts().nlargest(5).index.tolist()
                detail += f", Top Values: {top_values}"
            elif unique_count > 10 and dtype == 'object':
                 sample_values = list(col_series.dropna().sample(min(5, unique_count)))
                 detail += f", Sample Values: {sample_values}"

            if dtype == 'object':
                numeric_elements = pd.to_numeric(col_series, errors='coerce').notna().sum()
                if numeric_elements > 0 and numeric_elements < non_null_count:
                    detail += " **[WARNING: Mixed data types detected (numeric and text)]**"
            
            column_details.append(detail)

        sample_rows_md = df.head(5).to_markdown(index=False)

        prompt = ChatPromptTemplate.from_messages([("system", SUMMARY_REPORT_PROMPT), ("human", "Generate the detailed summary report for the dataframe.")]).format_messages(
            df_name=df_name,
            df_shape=df_info['shape'],
            df_memory_mb=df_info['memory_usage'] / 1024**2,
            has_nulls_text="Yes" if df_info['has_nulls'] else "No",
            column_details="\n".join(column_details),
            sample_rows=sample_rows_md
        )
        
        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            return f"Error generating summary report: {str(e)}"

    def chat(self, query: str) -> Dict[str, Any]:
        if not self.data_manager.dataframes:
            return {"error": "No Excel files loaded.", "result": None, "code": None, "result_type": "text", "raw_result": None}
        
        initial_state = ExcelChatState(
            messages=[], user_query=query, generated_code="", execution_result=None,
            error_message="", dataframe_info={}, retry_count=0, final_answer="",
            result_type="text", raw_result=None, validity=""
        )
        
        try:
            final_state = self.graph.invoke(initial_state)
            
            if "PROCEED" not in final_state.get("validity", ""):
                return {"error": None, "result": final_state.get("final_answer"), "code": None, "needs_clarification": False, "result_type": "text", "raw_result": None}

            if final_state["error_message"] and final_state.get("retry_count", 0) >= 2:
                return {"error": None, "result": final_state.get("final_answer", "Please rephrase your query."), "code": final_state["generated_code"], "needs_clarification": True, "result_type": "text", "raw_result": None}
            elif final_state["error_message"]:
                return {"error": final_state["error_message"], "result": None, "code": final_state["generated_code"], "needs_clarification": False, "result_type": "text", "raw_result": None}
            else:
                result_to_return = final_state.get("final_answer") or final_state["execution_result"]
                return {"error": None, "result": result_to_return, "code": final_state["generated_code"], "needs_clarification": False, "result_type": final_state.get("result_type", "text"), "raw_result": final_state.get("raw_result")}
        except Exception as e:
            return {"error": f"System error: {str(e)}", "result": None, "code": None, "needs_clarification": False, "result_type": "text", "raw_result": None}
