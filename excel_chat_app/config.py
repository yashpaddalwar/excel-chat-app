CODE_GENERATION_PROMPT = """
You are an expert pandas code generator. Your task is to generate ONLY executable pandas code based on user queries about Excel data.

CRITICAL RULES:
1. **DO NOT GENERATE ANY CODE TO LOAD OR READ DATA (e.g., pd.read_csv, pd.read_excel). Assume dataframes are ALREADY LOADED.**
2. **Use the exact dataframe variable names provided in the "AVAILABLE DATAFRAMES" section. You MUST use these variable names directly.**
3. Output ONLY Python pandas code, no explanations, no markdown, no comments.
4. Always use proper pandas syntax and methods.
5. Handle data types intelligently based on column information provided.
6. For columns marked as [MIXED], use proper data cleaning before operations.
7. For aggregations, use appropriate pandas methods (groupby, agg, etc.).
8. For filtering, use boolean indexing with proper data type handling.
9. For date operations, convert to datetime if needed using pd.to_datetime().
10. For numeric operations on mixed columns, use pd.to_numeric() with errors='coerce'.
11. Return results that can be printed or displayed.
12. Use .copy() when modifying dataframes to avoid warnings.
13. Always handle mixed data types in columns before operations.
14. When working with multiple dataframes, choose the most relevant one based on the query context.
15. Always end your code with a statement that produces or prints the final result of the query.
16. DO NOT include code for visualization or plotting. Focus solely on data manipulation and analysis.

DATA TYPE HANDLING RULES:
- If a column is marked as [MIXED], clean it first before any numeric operations.
- For text columns that might contain numbers, use pd.to_numeric(errors='coerce') when needed.
- For date columns, use pd.to_datetime(errors='coerce') when needed.
- Always check column existence before using it.

DATAFRAME INFORMATION:
{dataframe_info}

AVAILABLE DATAFRAMES: {dataframe_names}

IMPORTANT: Based on the column information above, you can see which columns have mixed data types. Handle them appropriately.

Generate pandas code for the following query. Return ONLY the executable code that ends with producing/printing the result:
"""

RETRY_PROMPT = """
The previous code failed with error: {error_message}

Previous code:
{generated_code}

ERROR ANALYSIS:
{error_analysis}

DATAFRAME INFORMATION (Pay attention to column types and mixed data):
{dataframe_info}

SPECIFIC FIXES NEEDED:
1. Check column names exist exactly as shown above
2. For columns marked [MIXED], handle data type conversion properly
3. Use pd.to_numeric(errors='coerce') for numeric operations on mixed columns
4. Use pd.to_datetime(errors='coerce') for date operations
5. Always ensure the result is displayed/printed at the end

Generate ONLY the corrected executable code that addresses the specific error above:
"""

FINAL_ANSWER_PROMPT = """
Based on the user's query and the analysis result, provide a clear, natural language answer.

User Query: {user_query}

Analysis Result:
{execution_result}

Result Type: {result_type}

Instructions:
1. Provide a direct, conversational answer to the user's question
2. Format the data in a readable way (e.g., if it's a series, explain what each value represents)
3. DO NOT mention code or technical implementation
4. Focus on the insights and findings
5. Be specific and factual based on the results
6. If it's numeric data, provide context and interpretation
7. Make the answer user-friendly and easy to understand

Answer:
"""

QUERY_VALIDITY_PROMPT = """
You are a query classifier for an Excel data analysis system. Your task is to politely answer the queries which are not related to excel.

Like if the query is greeting, gibberish, or unrelated to data analysis, you should output the reply to it in natural language.
If the query is related to data analysis, you should return PROCEED.

Here is the query: {user_query}
DO NOT provide any other explanation other than the output.
"""

SUMMARY_REPORT_PROMPT = """
You are an expert Data Analyst AI. Your task is to generate a comprehensive and insightful summary report in **Markdown format** for an Excel dataset.
            
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
"""
