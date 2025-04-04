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

from app import TextToSqlAgent

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

# --- Agent Initialization ---
# Agent instance - initialized only after data is loaded
agent: Optional[TextToSqlAgent] = None

# --- Helper Functions ---
def cleanup_old_dirs():
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
            except Exception as e:
                # Log warning but don't stop startup if cleanup fails
                logging.warning(f"Could not clean up directory {dir_to_delete} at startup: {e}")

    # Always ensure UPLOAD_FOLDER exists after cleanup attempt
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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
            except Exception as e:
                # Log warning but don't stop startup if cleanup fails
                logging.warning(f"Could not clean up directory {dir_to_delete} at startup: {e}")

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
            agent = TextToSqlAgent(db_path=DB_PATH, cache_file=SCHEMA_CACHE_FILE, default_model=DEFAULT_LLM_MODEL)
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
    
    # Delete DB
    if os.path.exists(DB_PATH):
        try:
            os.remove(DB_PATH)
            deleted_files.append(DB_PATH)
            logging.info(f"Deleted database: {DB_PATH}")
        except Exception as e:
            logging.error(f"Error deleting database {DB_PATH}: {e}")
            errors.append(f"DB deletion: {e}")
            
    # Delete Schema Cache
    if os.path.exists(SCHEMA_CACHE_FILE):
        try:
            os.remove(SCHEMA_CACHE_FILE)
            deleted_files.append(SCHEMA_CACHE_FILE)
            logging.info(f"Deleted schema cache: {SCHEMA_CACHE_FILE}")
        except Exception as e:
            logging.error(f"Error deleting schema cache {SCHEMA_CACHE_FILE}: {e}")
            errors.append(f"Schema cache deletion: {e}")
            
    # No longer need long initial sleep here, rename is faster
    
    # Directory deletion is now handled by cleanup_cache_dirs on startup
            
    return deleted_files, errors

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
    """Returns the current status of the agent (initialized or not)."""
    is_ready = agent is not None and agent.is_initialized
    return jsonify({"is_ready": is_ready})

@app.route('/query', methods=['POST'])
def handle_query():
    """Handles user queries sent from the frontend."""
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
    """Handles CSV file uploads, converts to SQLite, replaces old data, triggers indexing."""
    if 'files' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
        
    files = request.files.getlist('files')
    
    if not files or files[0].filename == '':
        return jsonify({"error": "No selected files"}), 400

    # 1. Delete existing data and caches first (replace workflow)
    delete_existing_data_and_caches()
    
    # 2. Save uploaded files temporarily
    saved_files = []
    for file in files:
        if file and file.filename.lower().endswith('.csv'):
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            saved_files.append(filepath)
            logging.info(f"Saved uploaded file: {filepath}")
        else:
            logging.warning(f"Skipping non-CSV file: {file.filename}")
            # Optionally return error if non-CSV is critical

    if not saved_files:
        return jsonify({"error": "No valid CSV files were uploaded."}), 400

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
    """Removes the database and all associated caches."""
    global agent
    logging.warning("Received request to remove data. Shutting down agent first...")
    if agent:
        agent.shutdown()
        agent = None
        gc.collect() # Suggest garbage collection
        logging.info("Agent shut down and global reference removed.")
    else:
        logging.info("No active agent instance to shut down.")
        
    # Now attempt deletion
    deleted_files, errors = delete_existing_data_and_caches()
    
    if errors:
         return jsonify({"message": f"Attempted to remove data. Files deleted: {deleted_files}. Errors encountered: {errors}"}), 500
    else:
        return jsonify({"message": f"Successfully removed data and caches: {deleted_files}"}), 200

# --- Refresh Endpoint ---
@app.route('/refresh-index', methods=['POST'])
def handle_refresh():
    """Handles request to refresh the schema cache and vector index."""
    if agent is None:
        return jsonify({"error": "Agent is not initialized. Cannot refresh index."}), 500

    logging.info("Received request to refresh index...")
    try:
        agent.refresh_index()
        logging.info("Index refresh process completed on agent.")
        return jsonify({"message": "Index refresh initiated and completed successfully."}), 200
    except Exception as e:
        logging.error(f"Error during index refresh: {e}", exc_info=True)
        return jsonify({"error": f"Failed to refresh index: {e}"}), 500

# --- Application Runner ---
if __name__ == '__main__':
    # Use 0.0.0.0 to make it accessible outside the container
    app.run(host='0.0.0.0', port=5000, debug=False) # Turn debug off for production 