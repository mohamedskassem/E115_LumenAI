import logging
from flask import Flask, request, jsonify, render_template
import os
import pandas as pd
import sqlite3
import shutil
import time # For adding delays
import errno # For checking specific OS errors
import gc # Garbage Collector
from typing import Optional
from flasgger import Swagger # Import Swagger

from app import TextToSqlAgent
from data_handler import _get_schema_cache_path

# --- Constants & Configuration ---
OLD_VECTOR_DIR_SUFFIX = '_old'
OLD_UPLOADS_DIR_SUFFIX = '_old'
UPLOAD_FOLDER = 'uploads' # Temp folder for uploads
DB_FOLDER = 'output'
DB_NAME = 'user_data.db'
DB_PATH = os.path.join(DB_FOLDER, DB_NAME)
SCHEMA_CACHE_FILE = 'schema_cache.json'
VECTOR_STORE_CACHE_DIR = 'vector_store_cache'
DEFAULT_LLM_MODEL = "gemini-2.5-pro-exp-03-25"

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DB_FOLDER, exist_ok=True)

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
swagger = Swagger(app) # Initialize Swagger

# --- Agent Initialization ---
# Agent instance - initialized only after data is loaded
agent: Optional[TextToSqlAgent] = None

# --- Helper Functions ---
def cleanup_old_dirs():
    """Deprecated: This function's purpose is now handled by cleanup_cache_dirs."""
    logging.warning("cleanup_old_dirs is deprecated and should not be called. Use cleanup_cache_dirs.")
    pass # Keep the function defined to avoid potential import errors if referenced elsewhere unexpectedly

def cleanup_cache_dirs():
    """Attempts to delete cache/upload directories from previous runs at startup."""
    logging.info("Checking for and cleaning up cache/upload directories from previous runs...")
    
    # Target the main directories for deletion at startup
    for dir_to_delete in [VECTOR_STORE_CACHE_DIR, UPLOAD_FOLDER]:
        if os.path.exists(dir_to_delete):
            try:
                # Add a small delay before attempting deletion
                time.sleep(0.2)
                shutil.rmtree(dir_to_delete)
                logging.info(f"Successfully cleaned up directory: {dir_to_delete}")
            except OSError as e:
                 # Specifically catch OSError which includes EBUSY
                 if e.errno == errno.EBUSY:
                      logging.warning(f"Could not clean up directory {dir_to_delete} at startup due to EBUSY (Device or resource busy). Skipping.")
                 else:
                      logging.warning(f"Could not clean up directory {dir_to_delete} at startup due to OS error: {e}")
            except Exception as e:
                # Log warning but don't stop startup if cleanup fails for other reasons
                logging.warning(f"Could not clean up directory {dir_to_delete} at startup due to unexpected error: {e}")

    # Always ensure UPLOAD_FOLDER exists after cleanup attempt
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def initialize_agent_if_needed():
    global agent
    if agent and agent.is_initialized:
        return True # Already initialized
        
    if os.path.exists(DB_PATH):
        logging.info(f"Database found at {DB_PATH}. Initializing agent...")
        try:
            # Create a new agent instance or re-initialize existing one
            if agent: agent.shutdown() # Ensure old resources are released if re-initializing
            agent = TextToSqlAgent(db_path=DB_PATH, default_model=DEFAULT_LLM_MODEL)
            if agent.load_and_index(): # Trigger the loading and indexing process
                logging.info("Agent initialized successfully.")
                return True
            else:
                logging.error("Agent initialization failed after load_and_index call.")
                agent = None # Failed
                return False
        except Exception as e:
            logging.error(f"Exception during agent initialization: {e}", exc_info=True)
            agent = None
            return False
    else:
        logging.info("Database not found. Agent cannot be initialized yet.")
        if agent: agent.shutdown() # Shutdown if DB was removed
        agent = None
        return False

def delete_existing_data_and_caches():
    global agent
    logging.warning("Deleting existing data and caches...")
    if agent:
        agent.shutdown()
        agent = None
        
    deleted_files = []
    errors = []
    ebusy_warnings = [] # Track EBUSY specifically
    
    # Delete DB
    if os.path.exists(DB_PATH):
        try:
            os.remove(DB_PATH)
            deleted_files.append(DB_PATH)
            logging.info(f"Deleted database: {DB_PATH}")
        except Exception as e:
            logging.error(f"Error deleting database {DB_PATH}: {e}")
            errors.append(f"DB deletion: {e}")
            
    # Delete Schema Cache File (if it exists) corresponding to the deleted DB
    # Get the expected cache file path using the helper from data_handler
    try:
        schema_cache_file_path = _get_schema_cache_path(DB_PATH) # DB_PATH is the path to the db that *was* deleted
        if os.path.exists(schema_cache_file_path):
            try:
                os.remove(schema_cache_file_path)
                deleted_files.append(schema_cache_file_path)
                logging.info(f"Deleted corresponding schema cache file: {schema_cache_file_path}")
            except Exception as e:
                logging.error(f"Error deleting schema cache file {schema_cache_file_path}: {e}")
                errors.append(f"Schema cache file deletion: {e}")
        else:
            logging.info(f"Schema cache file {schema_cache_file_path} not found, nothing to delete.")
    except ImportError:
        logging.error("Could not import _get_schema_cache_path from data_handler to delete cache file.")
        errors.append("Failed to import helper for schema cache deletion.")
    except Exception as e:
        logging.error(f"Error determining schema cache file path for deletion: {e}")
        errors.append(f"Error determining schema cache path: {e}")

    # --- Delete Vector Store Cache (Handle EBUSY gracefully) ---
    logging.info(f"Attempting to delete vector store cache: {VECTOR_STORE_CACHE_DIR}")
    if os.path.exists(VECTOR_STORE_CACHE_DIR):
        try:
            shutil.rmtree(VECTOR_STORE_CACHE_DIR)
            deleted_files.append(f"{VECTOR_STORE_CACHE_DIR} (directory)") # Indicate it's a directory
            logging.info("Vector store cache directory deleted.")
        except OSError as e:
             if e.errno == errno.EBUSY:
                  # Log as warning, don't treat as fatal error for this function's purpose
                  warning_msg = f"Could not delete vector store cache {VECTOR_STORE_CACHE_DIR} due to EBUSY (Device or resource busy). It might require manual removal or stopping/restarting Docker completely."
                  logging.warning(warning_msg)
                  ebusy_warnings.append(warning_msg)
                  # DO NOT add to errors list
             else:
                  # Treat other OS errors as errors
                  err_msg = f"Error deleting vector store cache {VECTOR_STORE_CACHE_DIR} (OS Error): {e}"
                  logging.error(err_msg)
                  errors.append(err_msg)
        except Exception as e:
            # Treat other unexpected errors as errors
            err_msg = f"Error deleting vector store cache {VECTOR_STORE_CACHE_DIR} (Unexpected Error): {e}"
            logging.error(err_msg)
            errors.append(err_msg)
    # -----------------------------------------------------------
            
    # Return deleted files, fatal errors, and non-fatal EBUSY warnings
    return deleted_files, errors, ebusy_warnings

# Try to initialize agent on startup if DB exists
# --- Perform initial cleanup and initialization ---
cleanup_cache_dirs() # Attempt cleanup right at the start
initialize_agent_if_needed()

# --- Routes ---
@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/status', methods=['GET'])
def get_status():
    """Returns the current status of the agent (initialized or not).
    ---
    tags:
      - Status
    responses:
      200:
        description: Agent status.
        schema:
          id: StatusResponse
          properties:
            is_ready:
              type: boolean
              description: True if the agent is initialized and ready, False otherwise.
    """
    is_ready = agent is not None and agent.is_initialized
    return jsonify({"is_ready": is_ready})

@app.route('/query', methods=['POST'])
def handle_query():
    """Handles user queries sent from the frontend.
    Receives a question and optionally a model name, processes it using the TextToSqlAgent,
    and returns the agent's response.
    ---
    tags:
      - Query
    parameters:
      - in: body
        name: body
        required: true
        schema:
          id: QueryRequest
          required:
            - question
          properties:
            question:
              type: string
              description: The user's natural language question.
            model:
              type: string
              description: (Optional) The specific LLM model name to use (e.g., 'gpt-4-turbo', 'models/gemini-pro'). Defaults to the server's configured default.
    responses:
      200:
        description: The agent's response to the query.
        schema:
          id: QueryResponse
          properties:
            type:
              type: string
              description: The type of response (e.g., 'sql_analysis', 'direct_answer', 'clarification_needed', 'error').
            message:
              type: string
              description: The natural language response or message from the agent.
            sql_query:
              type: string
              description: The SQL query generated by the agent, if applicable.
              nullable: true
      400:
        description: Bad request (e.g., agent not ready, no question provided).
      500:
        description: Internal server error during query processing.
    """
    # Ensure agent is ready before processing query
    if not initialize_agent_if_needed():
         return jsonify({"error": "Agent is not ready. Please load data first."}), 400 # Bad Request - client needs to load data
    # Agent should be non-None here if initialize_agent_if_needed returned True
    if not agent:
         # This case should ideally not happen if logic is correct, but safety check
         logging.error("Agent is None even after successful initialize check.")
         return jsonify({"error": "Internal server error: Agent state inconsistent."}), 500

    try:
        data = request.get_json()
        user_question = data.get('question')
        # Get model name from request, fallback to default if not provided
        model_name = data.get('model', DEFAULT_LLM_MODEL) # Use constant

        if not user_question:
            logging.warning("Received empty question.")
            return jsonify({"error": "No question provided."}), 400

        logging.info(f"Received query request for model '{model_name}': '{user_question}'")

        # Process the query using the agent
        response = agent.process_query(user_question, model_name)

        logging.info(f"Sending response: Type={response.get('type')}, Message snippet='{response.get('message', '')[:50]}...'")

        # Return the structured response
        return jsonify({
            "response_type": response.get("type"),
            "message": response.get("message"),
            "sql_query": response.get("sql_query") # Include SQL if generated
        })

    except Exception as e:
        logging.error(f"Error handling /query request: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred."}), 500

# --- Data Management Endpoints ---
@app.route('/upload-data', methods=['POST'])
def upload_data():
    """Handles CSV file uploads, converts to SQLite, replaces old data, triggers indexing.
    Accepts one or more CSV files.
    ---
    tags:
      - Data Management
    consumes:
      - multipart/form-data
    parameters:
      - name: files
        in: formData
        type: file
        required: true
        description: One or more CSV files to upload.
    responses:
      200:
        description: Files uploaded and agent initialized successfully.
        schema:
          id: UploadSuccess
          properties:
            message:
              type: string
      400:
        description: Bad request (e.g., no files, no valid CSVs).
      500:
        description: Internal server error during file processing or agent initialization.
    """
    if 'files' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
        
    files = request.files.getlist('files')
    
    if not files or files[0].filename == '':
        return jsonify({"error": "No selected files"}), 400

    # 1. Delete existing data and caches first (replace workflow)
    # Ignore ebusy_warnings from this step for now, focus on fatal errors
    _, delete_errors, _ = delete_existing_data_and_caches()
    if delete_errors:
         # If critical deletions failed (DB/Schema cache), maybe stop?
         # For now, let's log and continue, but report later if init fails.
         logging.error(f"Errors occurred during pre-upload cleanup: {delete_errors}. Proceeding with upload, but initialization might fail.")

    
    # 2. Save uploaded files temporarily
    saved_files = []
    for file in files:
        if file and file.filename.lower().endswith('.csv'):
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            try:
                file.save(filepath)
                saved_files.append(filepath)
                logging.info(f"Saved uploaded file: {filepath}")
            except Exception as e:
                 logging.error(f"Error saving uploaded file {file.filename} to {filepath}: {e}")
                 return jsonify({"error": f"Failed to save uploaded file {file.filename}"}), 500
        else:
            logging.warning(f"Skipping non-CSV file: {file.filename}")
            # Optionally return error if non-CSV is critical

    if not saved_files:
        return jsonify({"error": "No valid CSV files were uploaded or saved."}), 400

    # 3. Convert CSVs to SQLite DB
    try:
        conn = sqlite3.connect(DB_PATH)
        logging.info(f"Creating/Replacing database: {DB_PATH}")
        for csv_path in saved_files:
            table_name = os.path.splitext(os.path.basename(csv_path))[0]
            # Sanitize table name (basic example, might need more robust logic)
            table_name = "".join(c if c.isalnum() else '_' for c in table_name)
            if not table_name: table_name = "data_table"
            
            logging.info(f"Processing {csv_path} into table '{table_name}'")

            # Try reading with different encodings
            df = None
            encodings_to_try = ['utf-8', 'cp1252', 'latin1']
            for encoding in encodings_to_try:
                try:
                    df = pd.read_csv(csv_path, encoding=encoding)
                    logging.info(f"Successfully read {csv_path} using encoding '{encoding}'")
                    break  # Success, stop trying encodings
                except UnicodeDecodeError:
                    logging.warning(f"Failed to read {csv_path} with encoding '{encoding}'. Trying next...")
                except Exception as e:
                    # Handle other potential read errors (e.g., file corruption, pandas issues)
                    logging.error(f"Error reading {csv_path} with encoding '{encoding}': {e}")
                    df = None # Ensure df is None if error occurred
                    break # Stop trying encodings for this file on non-Unicode errors

            # Check if file was successfully read
            if df is None:
                logging.error(f"Failed to read {csv_path} with any attempted encoding. Skipping this file.")
                # Consider removing the problematic file from uploads folder if desired
                # try: os.remove(csv_path) except OSError: pass 
                continue # Move to the next file in the saved_files list

            # Convert column names to be SQL-friendly (basic example)
            df.columns = ["".join(c if c.isalnum() else '_' for c in str(col)).strip('_') for col in df.columns]
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            logging.info(f"Successfully loaded data into table '{table_name}'")
        conn.close()
    except Exception as e:
        logging.error(f"Error converting CSV to SQLite: {e}", exc_info=True)
        # Clean up potentially partial DB? Or leave it for inspection?
        # For now, just report error
        return jsonify({"error": f"Error processing CSV files: {e}"}), 500
        
    # 4. Trigger agent initialization/indexing
    if initialize_agent_if_needed():
        return jsonify({"message": f"Successfully uploaded {len(saved_files)} CSV(s) and initialized agent."}), 200
    else:
        return jsonify({"error": "Data loaded to DB, but agent initialization failed. Check logs."}), 500

@app.route('/remove-data', methods=['POST'])
def remove_data():
    """Removes the database and all associated caches.
    Shuts down the agent if running.
    Handles EBUSY errors during vector cache deletion gracefully.
    ---
    tags:
      - Data Management
    responses:
      200:
        description: Data and caches removed successfully (or cache removal skipped due to EBUSY).
        schema:
          id: RemoveSuccess
          properties:
            message:
              type: string
            warnings: 
              type: array
              description: List of non-fatal warnings (e.g., EBUSY during cache cleanup).
              items:
                type: string
      500:
        description: Critical errors occurred during cleanup (details may be in the message).
        schema:
          id: RemoveError
          properties:
            message:
              type: string
            errors: 
              type: array
              items:
                 type: string
    """
    global agent
    logging.warning("Received request to remove data. Shutting down agent first...")
    if agent:
        agent.shutdown()
        agent = None
        gc.collect() # Suggest garbage collection
        logging.info("Agent shut down and global reference removed.")
    else:
        logging.info("No active agent instance to shut down.")
        
    # Now attempt deletion, capturing errors and EBUSY warnings separately
    deleted_files, errors, ebusy_warnings = delete_existing_data_and_caches()
    
    if errors:
         # If critical errors occurred (DB/schema cache deletion failed)
         return jsonify({"message": f"Attempted to remove data. Files deleted: {deleted_files}. Critical errors encountered: {errors}", "errors": errors}), 500
    else:
        # If only EBUSY warnings occurred, or no errors/warnings
        success_message = f"Successfully removed critical data (DB, schema cache): {deleted_files}."
        if ebusy_warnings:
             success_message += " Warning: Vector cache directory may still exist due to 'Device or resource busy' error and might require manual cleanup."
        return jsonify({"message": success_message, "warnings": ebusy_warnings}), 200

# --- Refresh Endpoint ---
@app.route('/refresh-index', methods=['POST'])
def handle_refresh():
    """Handles request to FORCE refresh of schema cache and vector index.
    Deletes existing caches and re-initializes the agent using existing DB data.
    This forces schema re-analysis and index rebuild. Use this after code changes affecting indexing or if caches are suspected to be stale.
    --- 
    tags:
      - Data Management
    responses:
      200:
        description: Agent cache cleared and re-initialized successfully.
        schema:
          id: RefreshSuccess
          properties:
            message:
              type: string
      500:
        description: Error during cache deletion or agent failed to re-initialize.
        schema:
          id: RefreshError
          properties:
            error:
              type: string
    """
    global agent # Ensure we modify the global agent

    logging.warning("Received request to FORCE refresh/rebuild of agent index...")

    # 1. Shutdown existing agent (important before deleting its potential cache files)
    if agent:
        logging.info("Shutting down existing agent before cache deletion...")
        agent.shutdown()
        agent = None
        gc.collect() # Suggest cleanup

    # 2. Delete Caches
    errors = []
    logging.info(f"Attempting to delete schema cache: {SCHEMA_CACHE_FILE}")
    if os.path.exists(SCHEMA_CACHE_FILE):
        try:
            os.remove(SCHEMA_CACHE_FILE)
            logging.info("Schema cache deleted.")
        except Exception as e:
            logging.error(f"Error deleting schema cache {SCHEMA_CACHE_FILE}: {e}")
            errors.append(f"Schema cache deletion error: {e}")

    logging.info(f"Attempting to delete vector store cache: {VECTOR_STORE_CACHE_DIR}")
    if os.path.exists(VECTOR_STORE_CACHE_DIR):
        try:
            shutil.rmtree(VECTOR_STORE_CACHE_DIR)
            logging.info("Vector store cache directory deleted.")
            # Recreate the directory immediately as load_and_index might expect it
            os.makedirs(VECTOR_STORE_CACHE_DIR, exist_ok=True)
        except Exception as e:
            logging.error(f"Error deleting vector store cache {VECTOR_STORE_CACHE_DIR}: {e}")
            errors.append(f"Vector store cache deletion error: {e}")

    if errors:
         logging.warning(f"Completed cache deletion attempt with errors: {errors}")
         # Decide whether to stop or proceed despite errors. Let's proceed.

    # 3. Re-initialize Agent
    logging.info("Attempting to re-initialize agent after cache deletion...")
    if initialize_agent_if_needed():
        # Check if errors occurred during deletion and report if so
        if errors:
             msg = f"Agent re-initialized, but errors occurred during cache cleanup: {errors}"
             logging.warning(msg)
             # Return 200 but indicate partial success in message
             return jsonify({"message": msg}), 200 
        else:
             logging.info("Agent re-initialized successfully after cache deletion.")
             return jsonify({"message": "Agent cache cleared and re-initialized successfully."}), 200
    else:
        error_msg = "Failed to re-initialize agent after deleting caches. Check database presence and logs."
        logging.error(error_msg)
        return jsonify({"error": error_msg}), 500

# --- Application Runner ---
if __name__ == '__main__':
    # Use 0.0.0.0 to make it accessible outside the container
    app.run(host='0.0.0.0', port=5000, debug=False) # Turn debug off for production 