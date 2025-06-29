import os
import pandas as pd
from typing import Dict, Optional
from excel_chat_app.logger_config import logger

class DataManager:
    """
    Handles loading, storing, and analyzing Excel dataframes.
    """
    def __init__(self):
        self.dataframes = {}
        logger.info("DataManager initialized.")

    def load_excel_file(self, file_path: str, original_filename: str = None, sheet_name: Optional[str] = None) -> str:
        logger.info(f"Loading Excel file: {original_filename or os.path.basename(file_path)}")
        """
        Loads an Excel file into a pandas DataFrame.

        Args:
            file_path (str): The path to the Excel file.
            original_filename (str, optional): The original name of the uploaded file. Defaults to None.
            sheet_name (str, optional): The name of the sheet to load. Defaults to None.

        Returns:
            str: The name assigned to the loaded dataframe.
        """
        try:
            if sheet_name:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                base_name = os.path.splitext(original_filename or os.path.basename(file_path))[0]
                df_name = f"{base_name}_{sheet_name}"
            else:
                df = pd.read_excel(file_path)
                df_name = os.path.splitext(original_filename or os.path.basename(file_path))[0]

            # Clean column names
            df.columns = df.columns.str.strip()
            
            self.dataframes[df_name] = df
            logger.info(f"Loaded dataframe '{df_name}' with shape: {df.shape}")
            return df_name
        except Exception as e:
            logger.error(f"Error loading Excel file '{original_filename}': {e}", exc_info=True)
            raise

    def get_dataframe_info(self) -> Dict:
        """
        Gathers comprehensive information about the loaded dataframes.

        Returns:
            Dict: A dictionary containing detailed information for each dataframe.
        """
        info = {}
        for name, df in self.dataframes.items():
            sample_data = df.head(3).to_string(index=False)
            
            col_info = []
            for col in df.columns:
                dtype = str(df[col].dtype)
                non_null = df[col].count()
                unique_vals = df[col].nunique()
                
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

    def get_simple_dataframe_summary(self) -> str:
        """
        Returns a simple summary of the loaded dataframes.

        Returns:
            str: A string containing the names and columns of the loaded dataframes.
        """
        summary = ""
        for name, df in self.dataframes.items():
            summary += f"\n{name}: {list(df.columns)}"
        return summary
