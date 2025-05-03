import os
import sqlite3
import json
import logging
import concurrent.futures
from typing import List, Tuple, Dict, Optional, Union

import google.generativeai as genai

# Assuming necessary LlamaIndex components are available if needed
# We need Document for return type, OpenAI for type hint, PromptTemplate for LLM call
from llama_index.core import Document, PromptTemplate
from llama_index.llms.openai import OpenAI

# NEW: Define a path for the schema analysis cache
SCHEMA_ANALYSIS_CACHE_DIR = "schema_analysis_cache"


def _get_schema_cache_path(db_path: str) -> str:
    """Generates a unique cache file path based on the database filename."""
    db_filename = os.path.basename(db_path)
    cache_filename = f"{db_filename}.json"
    return os.path.join(SCHEMA_ANALYSIS_CACHE_DIR, cache_filename)


def _analyze_column(
    col_name,
    table_name,
    sample_data,
    llm: Optional[Union[OpenAI, genai.GenerativeModel]],
):
    """Analyzes a single column using the provided LLM instance."""
    # Handle case where LLM initialization might have failed
    if llm is None:
        logging.warning(
            f"LLM not available for analyzing column {table_name}.{col_name}. Returning basic info."
        )
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
        response_text = "(LLM analysis not performed)"
        if isinstance(llm, OpenAI):
            # LlamaIndex OpenAI predict for schema analysis seems to require PromptTemplate
            response = llm.predict(PromptTemplate(template=combined_prompt))
            response_text = response.strip()
        elif isinstance(llm, genai.GenerativeModel):
            # Use google-generativeai
            response = llm.generate_content(combined_prompt)
            # Handle potential safety blocks or errors
            if response.parts:
                response_text = response.text.strip()  # Access text via .text
            else:
                logging.warning(
                    f"GenAI response for {table_name}.{col_name} blocked or empty. Reason: {response.prompt_feedback}"
                )
                response_text = "(LLM analysis blocked or empty)"

        doc_text = (
            f"Table: {table_name}\n" f"Column: {col_name}\n" f"Summary: {response_text}"
        )
        # Store raw analysis text and structured info
        return {
            "text": doc_text,
            "table": table_name,
            "column": col_name,
            "analysis": response_text,
        }
    except Exception as e:
        logging.error(
            f"Error analyzing column {table_name}.{col_name} with LLM {type(llm)}: {e}",
            exc_info=True,
        )
        # Return basic info even if LLM fails
        basic_text = f"Table: {table_name}\nColumn: {col_name}\nSummary: Analysis failed due to error."
        return {
            "text": basic_text,
            "table": table_name,
            "column": col_name,
            "analysis": "(Analysis failed due to error)",
        }


def _process_table(
    table_info: Tuple[str, List[Tuple], List[Tuple], dict],
    llm: Optional[Union[OpenAI, genai.GenerativeModel]],
) -> Tuple[str, List[Dict]]:
    """Processes a single table and its columns."""
    table_name, columns, sample_data, column_indices = table_info
    table_schema = f"Table: {table_name}\n"
    analysis_results = []  # Store analysis dicts
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
        analysis_result = _analyze_column(col_name, table_name, col_samples, llm)
        analysis_results.append(analysis_result)

    table_schema += "\n"
    return table_schema, analysis_results


def load_db_schema_and_analysis(
    db_path: str,
    analysis_llm: Optional[Union[OpenAI, genai.GenerativeModel]],
    force_regenerate: bool = False,
) -> Tuple[str, List[Document], Optional[Dict]]:
    """Loads database schema and detailed analysis, using or generating cache.

    Args:
        db_path: Path to the SQLite database file.
        analysis_llm: An initialized LLM instance (OpenAI or genai.GenerativeModel) or None.
        force_regenerate: If True, ignores existing cache and regenerates analysis.

    Returns:
        A tuple containing:
            - full_schema: The raw schema string.
            - schema_documents: List of LlamaIndex Documents for vector indexing (may be basic if analysis failed).
            - detailed_analysis: Dictionary containing structured LLM analysis (or None if failed/skipped/not cached).
    """
    cache_file = _get_schema_cache_path(db_path)
    full_schema = ""
    schema_documents = []
    detailed_analysis = None

    # Attempt to load from cache first, unless forced to regenerate
    if not force_regenerate and os.path.exists(cache_file):
        logging.info(f"Loading cached schema analysis from {cache_file}...")
        try:
            with open(cache_file, "r") as f:
                cache = json.load(f)
            # Validate cache structure (simple check)
            if (
                "raw_schema" in cache
                and "analysis_results" in cache
                and isinstance(cache["analysis_results"], list)
            ):
                full_schema = cache["raw_schema"]
                detailed_analysis = {
                    f"{item['table']}.{item['column']}": item["analysis"]
                    for item in cache["analysis_results"]
                }
                # Recreate Documents from cached text for vector store
                schema_documents = [
                    Document(text=item["text"])
                    for item in cache["analysis_results"]
                    if "text" in item
                ]
                logging.info(
                    f"Successfully loaded schema and analysis from cache: {cache_file}"
                )
                # No DB connection needed if loading from cache
                return full_schema, schema_documents, detailed_analysis
            else:
                logging.warning(
                    f"Cache file {cache_file} has invalid format. Re-generating."
                )
        except (json.JSONDecodeError, KeyError, Exception) as e:
            logging.warning(
                f"Failed to load or parse cache file {cache_file}: {e}. Re-generating schema.",
                exc_info=True,
            )
            # If cache is invalid, proceed to generate fresh schema

    # --- Proceed with DB connection and analysis if cache failed, absent, or forced ---
    conn = None
    all_analysis_results = []  # Collect results from all tables

    try:
        logging.info(
            f"Connecting to database {db_path} for schema analysis (Regenerating: {force_regenerate})..."
        )
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        llm_type_str = "No LLM"
        if isinstance(analysis_llm, OpenAI):
            llm_type_str = f"OpenAI ({analysis_llm.model})"
        elif isinstance(analysis_llm, genai.GenerativeModel):
            # genai model name access might differ, adjust if needed
            # Using a placeholder for now
            llm_type_str = f"Google GenAI ({analysis_llm._model_name})"

        logging.info(
            f"Found {len(tables)} tables. Analyzing schema (using {llm_type_str})..."
        )

        table_data = []
        for table in tables:
            table_name = table[0]
            try:
                cursor.execute(f'PRAGMA table_info("{table_name}");')
                columns = cursor.fetchall()
                cursor.execute(f'SELECT * FROM "{table_name}" LIMIT 50;')
                sample_data = cursor.fetchall()
                column_indices = {col[1]: idx for idx, col in enumerate(columns)}
                table_data.append((table_name, columns, sample_data, column_indices))
            except sqlite3.Error as e:
                logging.error(
                    f"Error fetching schema/data for table {table_name}: {e}",
                    exc_info=True,
                )

        # Use ThreadPoolExecutor for parallel processing (if LLM available)
        if analysis_llm and table_data:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=4
            ) as executor:  # Increased workers slightly
                future_to_table = {
                    executor.submit(
                        _process_table, table_info, analysis_llm
                    ): table_info[0]
                    for table_info in table_data
                }
                for future in concurrent.futures.as_completed(future_to_table):
                    table_name = future_to_table[future]
                    try:
                        table_schema, table_analysis_results = future.result()
                        full_schema += table_schema
                        all_analysis_results.extend(table_analysis_results)
                        logging.info(f"Completed processing table: {table_name}")
                    except Exception as e:
                        logging.error(
                            f"Error processing table {table_name} in thread: {e}",
                            exc_info=True,
                        )
        elif table_data:  # If no LLM, just generate basic schema info without analysis
            logging.warning(
                "LLM for analysis not provided or failed. Generating basic schema structure only."
            )
            for table_info in table_data:
                table_name, columns, _, _ = table_info
                table_schema_str = f"Table: {table_name}\n"
                basic_analysis_results = []
                for col in columns:
                    table_schema_str += f"  - {col[1]} ({col[2]})\n"
                    # Create basic analysis structure without LLM summary
                    basic_text = f"Table: {table_name}\nColumn: {col[1]}\nSummary: (Analysis not performed)"
                    basic_analysis_results.append(
                        {
                            "text": basic_text,
                            "table": table_name,
                            "column": col[1],
                            "analysis": "(Analysis not performed)",
                        }
                    )
                full_schema += table_schema_str + "\n"
                all_analysis_results.extend(basic_analysis_results)
            logging.info("Generated basic schema structure without LLM analysis.")

        # Prepare data for caching and return
        if full_schema and all_analysis_results:
            schema_documents = [
                Document(text=item["text"]) for item in all_analysis_results
            ]
            detailed_analysis = {
                f"{item['table']}.{item['column']}": item["analysis"]
                for item in all_analysis_results
            }
            try:
                cache_data = {
                    "raw_schema": full_schema,
                    "analysis_results": all_analysis_results,  # Store the list of dicts
                }
                # Ensure cache directory exists ONLY when saving
                os.makedirs(SCHEMA_ANALYSIS_CACHE_DIR, exist_ok=True)
                with open(cache_file, "w") as f:
                    json.dump(cache_data, f, indent=4)
                logging.info(
                    f"Schema analysis complete/attempted and saved to {cache_file}."
                )
            except IOError as e:
                logging.error(
                    f"Error saving cache file {cache_file}: {e}", exc_info=True
                )

    except sqlite3.Error as e:
        logging.error(f"Database error during schema loading: {e}", exc_info=True)
        return "", [], None
    except Exception as e:
        logging.error(f"Unexpected error during schema loading: {e}", exc_info=True)
        return "", [], None
    finally:
        if conn:
            conn.close()
            logging.info(
                f"Closed temporary DB connection for schema analysis: {db_path}"
            )

    return full_schema, schema_documents, detailed_analysis


def get_db_connection(
    db_path: str,
) -> Optional[Tuple[sqlite3.Connection, sqlite3.Cursor]]:
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
        logging.error(
            f"Error connecting to database {db_path} for persistent connection: {e}",
            exc_info=True,
        )
        return None
