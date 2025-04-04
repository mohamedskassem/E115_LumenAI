import os
import sqlite3
import json
import logging
import concurrent.futures
import time
from typing import List, Tuple, Dict, Optional

# Assuming necessary LlamaIndex components are available if needed
# We need Document for return type, OpenAI for type hint, PromptTemplate for LLM call
from llama_index.core import Document, PromptTemplate
from llama_index.llms.openai import OpenAI


def _analyze_column(col_name, table_name, sample_data, llm: Optional[OpenAI]):
    """Analyzes a single column using the provided LLM instance."""
    # Handle case where LLM initialization might have failed
    if llm is None:
        logging.warning(f"LLM not available for analyzing column {table_name}.{col_name}. Returning basic info.")
        # Provide a basic summary without LLM analysis
        return f"Table: {table_name}\nColumn: {col_name}\nSummary: (LLM analysis not available)"

    combined_prompt = f"""
    Briefly describe column '{col_name}' in table '{table_name}':
    - Purpose
    - Data type
    - Sample values: {sample_data[:3]}

    Keep the response under 2 sentences.
    """
    try:
        # Use .complete for newer LlamaIndex versions if .predict is deprecated
        # Assuming .predict is still valid for the version in use based on original code
        response = llm.predict(PromptTemplate(template=combined_prompt))
        # REMOVED time.sleep(1) - Let's keep delays out of the handler for now
        doc_text = (
            f"Table: {table_name}\n"
            f"Column: {col_name}\n"
            f"Summary: {response.strip()}"
        )
        return doc_text
    except Exception as e:
        logging.error(
            f"Error analyzing column {table_name}.{col_name} with LLM {type(llm)}: {e}", exc_info=True
        )
        # Return a basic doc text even if LLM fails
        return f"Table: {table_name}\nColumn: {col_name}\nSummary: Analysis failed due to error."

def _process_table(
    table_info: Tuple[str, List[Tuple], List[Tuple], dict], llm: Optional[OpenAI]
) -> Tuple[str, List[Document]]:
    """Processes a single table and its columns."""
    table_name, columns, sample_data, column_indices = table_info
    table_schema = f"Table: {table_name}\n"
    schema_documents = []
    logging.debug(f"Processing table: {table_name}")

    for col in columns:
        col_name = col[1]
        col_type = col[2]
        table_schema += f"  - {col_name} ({col_type})\n"

        col_samples = [
            row[column_indices[col_name]]
            for row in sample_data
            if len(row) > column_indices[col_name]
        ]

        # Pass the potentially None LLM instance
        doc_text = _analyze_column(col_name, table_name, col_samples, llm)
        schema_documents.append(Document(text=doc_text))

    table_schema += "\n"
    return table_schema, schema_documents

def load_db_schema_and_summaries(
    db_path: str, cache_file: str, analysis_llm: Optional[OpenAI]
) -> Tuple[str, List[Document]]:
    """Loads database schema and generates summaries with parallel processing and caching.
    
    Args:
        db_path: Path to the SQLite database file.
        cache_file: Path to the JSON file for caching schema analysis.
        analysis_llm: An initialized OpenAI LLM instance (or None if LLM analysis failed/skipped).
        
    Returns:
        A tuple containing the full schema string and a list of LlamaIndex Document objects 
        representing column summaries.
    """
    # Attempt to load from cache first
    if os.path.exists(cache_file):
        logging.info(f"Loading cached schema analysis from {cache_file}...")
        try:
            with open(cache_file, "r") as f:
                cache = json.load(f)
            # Validate cache structure (simple check)
            if "schema" in cache and "documents" in cache:
                 logging.info(f"Successfully loaded schema from cache: {cache_file}")
                 # No connection needed if loading from cache
                 return (
                     cache["schema"],
                     [Document(text=doc) for doc in cache["documents"]],
                 )
            else:
                 logging.warning(f"Cache file {cache_file} has invalid format. Re-generating.")
        except (json.JSONDecodeError, KeyError, Exception) as e:
            logging.warning(
                f"Failed to load or parse cache file {cache_file}: {e}. Re-generating schema.",
                exc_info=True,
            )
            # If cache is invalid, proceed to generate fresh schema

    # --- Proceed with DB connection and analysis if cache failed or absent ---
    conn = None
    full_schema = ""
    schema_documents = []
    
    try:
        logging.info(f"Connecting to database {db_path} for schema analysis...")
        conn = sqlite3.connect(db_path) # Use check_same_thread=False only if needed later
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        logging.info(f"Found {len(tables)} tables. Analyzing schema (using {analysis_llm.model if analysis_llm else 'No LLM'})...")

        table_data = []
        for table in tables:
            table_name = table[0]
            try:
                cursor.execute(f'PRAGMA table_info("{table_name}");')
                columns = cursor.fetchall()
                cursor.execute(f'SELECT * FROM "{table_name}" LIMIT 50;') # Limit sample data
                sample_data = cursor.fetchall()
                column_indices = {col[1]: idx for idx, col in enumerate(columns)}
                table_data.append((table_name, columns, sample_data, column_indices))
            except sqlite3.Error as e:
                logging.error(f"Error fetching schema/data for table {table_name}: {e}", exc_info=True)

        # Use ThreadPoolExecutor for parallel processing (if LLM available)
        if analysis_llm and table_data:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                future_to_table = {
                    executor.submit(_process_table, table_info, analysis_llm): table_info[0]
                    for table_info in table_data
                }
                for future in concurrent.futures.as_completed(future_to_table):
                    table_name = future_to_table[future]
                    try:
                        table_schema, table_docs = future.result()
                        full_schema += table_schema
                        schema_documents.extend(table_docs)
                        logging.info(f"Completed processing table: {table_name}")
                    except Exception as e:
                        logging.error(f"Error processing table {table_name} in thread: {e}", exc_info=True)
        elif table_data: # If no LLM, just generate basic schema info without analysis
             logging.warning("LLM for analysis not provided or failed. Generating basic schema structure only.")
             for table_info in table_data:
                  table_name, columns, _, _ = table_info
                  table_schema_str = f"Table: {table_name}\n"
                  for col in columns:
                       table_schema_str += f"  - {col[1]} ({col[2]})\n"
                  full_schema += table_schema_str + "\n"
                  # Create basic documents without LLM summary
                  for col in columns:
                       doc_text = f"Table: {table_name}\nColumn: {col[1]}\nSummary: (Analysis not performed)"
                       schema_documents.append(Document(text=doc_text))
             logging.info("Generated basic schema structure without LLM analysis.")

        # Save to cache if analysis was attempted (even if partially failed)
        if full_schema and schema_documents: 
            try:
                cache = {
                    "schema": full_schema,
                    "documents": [doc.text for doc in schema_documents],
                }
                with open(cache_file, "w") as f:
                    json.dump(cache, f, indent=4)
                logging.info(f"Schema analysis complete/attempted and saved to {cache_file}.")
            except IOError as e:
                logging.error(f"Error saving cache file {cache_file}: {e}", exc_info=True)

    except sqlite3.Error as e:
        logging.error(f"Database error during schema loading: {e}", exc_info=True)
        # Return empty structure on connection/query error
        return "", []
    except Exception as e:
        logging.error(f"Unexpected error during schema loading: {e}", exc_info=True)
        return "", []
    finally:
        if conn:
            conn.close()
            logging.info(f"Closed temporary DB connection for schema analysis: {db_path}")

    return full_schema, schema_documents

def get_db_connection(db_path: str) -> Optional[Tuple[sqlite3.Connection, sqlite3.Cursor]]:
    """Establishes a persistent connection to the SQLite database for query execution.
       Returns None if connection fails.
    """
    try:
        # check_same_thread=False is important for Flask if agent modifies DB
        # Or if agent is accessed across multiple requests without reinitialization
        conn = sqlite3.connect(db_path, check_same_thread=False) 
        cursor = conn.cursor()
        logging.info(f"Successfully established persistent DB connection: {db_path}")
        return conn, cursor
    except sqlite3.Error as e:
        logging.error(f"Error connecting to database {db_path} for persistent connection: {e}", exc_info=True)
        return None 