import logging
from flask import Flask, request, jsonify, render_template
import os
import pandas as pd
import sqlite3
import shutil
import time # For adding delays
import errno # For checking specific OS errors
import gc # Garbage Collector
from typing import Optional, Dict
import uuid # For generating unique chat IDs
from flasgger import Swagger # Import Swagger

from app import TextToSqlAgent
from data_handler import _get_schema_cache_path
from llm_interface import generate_chat_title

# --- Constants & Configuration ---
UPLOAD_FOLDER = 'uploads' # Temp folder for uploads
DB_FOLDER = 'output'
DB_NAME = 'user_data.db'
DB_PATH = os.path.join(DB_FOLDER, DB_NAME)
SCHEMA_CACHE_DIR = 'schema_analysis_cache' # Matches data_handler
VECTOR_STORE_CACHE_DIR = 'vector_store_cache'
DEFAULT_LLM_MODEL = "gemini-2.5-pro-preview-03-25"

# Ensure directories exist (Upload folder might be cleared later)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DB_FOLDER, exist_ok=True)
os.makedirs(SCHEMA_CACHE_DIR, exist_ok=True) # Ensure schema cache dir exists

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
swagger = Swagger(app) # Initialize Swagger

# --- Global State Management --- 
# Dictionary to hold active agent instances, keyed by chat_id
chat_agents: Dict[str, TextToSqlAgent] = {}

# Global variables to hold the shared loaded data components
shared_data_components: Optional[Dict[str, object]] = None
is_data_loaded: bool = False

# --- Helper Functions ---
def cleanup_cache_dirs(delete_db=False):
    """Attempts to delete cache/upload directories. Optionally deletes DB."""
    logging.info(f"Cleaning up cache/upload directories... (Delete DB: {delete_db})")
    global is_data_loaded, shared_data_components, chat_agents

    # 0. Shutdown existing agents before deleting caches/DB
    shutdown_all_agents()

    # 1. Target directories/files for deletion
    # Only delete vector store cache if deleting the DB
    dirs_to_delete = [UPLOAD_FOLDER]
    if delete_db:
        dirs_to_delete.append(VECTOR_STORE_CACHE_DIR)

    files_to_delete = []
    if delete_db and os.path.exists(DB_PATH):
        files_to_delete.append(DB_PATH)
        # Also try deleting the corresponding schema JSON cache if DB is deleted
        try:
            schema_cache_file_path = _get_schema_cache_path(DB_PATH)
            if os.path.exists(schema_cache_file_path):
                 files_to_delete.append(schema_cache_file_path)
        except Exception as e:
            logging.warning(f"Could not determine/add schema cache file for deletion: {e}")

    warnings = []
    errors = []

    # Delete directories
    for dir_path in dirs_to_delete:
        if os.path.exists(dir_path):
            try:
                time.sleep(0.1) # Short delay
                shutil.rmtree(dir_path)
                logging.info(f"Successfully cleaned up directory: {dir_path}")
            except OSError as e:
                 if e.errno == errno.EBUSY:
                      warn_msg = f"Could not clean up directory {dir_path} due to EBUSY. Skipping."
                      logging.warning(warn_msg)
                      warnings.append(warn_msg)
                 else:
                      err_msg = f"Could not clean up directory {dir_path} due to OS error: {e}"
                      logging.warning(err_msg)
                      errors.append(err_msg)
            except Exception as e:
                err_msg = f"Could not clean up directory {dir_path} due to unexpected error: {e}"
                logging.warning(err_msg)
                errors.append(err_msg)

    # Delete individual files (DB, specific schema cache)
    for file_path in files_to_delete:
         if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logging.info(f"Successfully deleted file: {file_path}")
            except Exception as e:
                 err_msg = f"Error deleting file {file_path}: {e}"
                 logging.error(err_msg)
                 errors.append(err_msg)

    # Always ensure necessary directories exist after cleanup
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(DB_FOLDER, exist_ok=True)
    os.makedirs(SCHEMA_CACHE_DIR, exist_ok=True)

    # Reset global state if DB was deleted
    if delete_db:
        is_data_loaded = False
        shared_data_components = None
        chat_agents = {}
        logging.info("Global data state reset because database was deleted.")

    return errors, warnings

def shutdown_all_agents():
    """Shuts down all active agent instances."""
    global chat_agents
    logging.info(f"Shutting down all ({len(chat_agents)}) active agent instances...")
    agent_ids = list(chat_agents.keys()) # Get keys before iterating
    for agent_id in agent_ids:
        agent = chat_agents.pop(agent_id, None)
        if agent:
            agent.shutdown()
    gc.collect()
    logging.info("Finished shutting down agents.")

def load_global_data(force_regenerate=False):
    """Loads data globally using the Agent's static method."""
    global shared_data_components, is_data_loaded
    logging.info(f"Attempting to load global data from DB: {DB_PATH} (Force regenerate: {force_regenerate})")
    if not os.path.exists(DB_PATH):
        logging.error("Cannot load global data: Database file not found.")
        is_data_loaded = False
        shared_data_components = None
        return False

    try:
        loaded_data = TextToSqlAgent.load_and_index_data(
            db_path=DB_PATH,
            persist_dir=VECTOR_STORE_CACHE_DIR,
            force_regenerate_analysis=force_regenerate
        )
        if loaded_data:
            shared_data_components = loaded_data
            is_data_loaded = True
            logging.info("Global data loaded and indexed successfully.")
            return True
        else:
            logging.error("Failed to load and index global data.")
            is_data_loaded = False
            shared_data_components = None
            return False
    except Exception as e:
        logging.error(f"Exception during global data load: {e}", exc_info=True)
        is_data_loaded = False
        shared_data_components = None
        return False

def create_new_chat_session() -> Optional[str]:
    """Creates a new agent instance for a new chat session."""
    global chat_agents, shared_data_components, is_data_loaded

    if not is_data_loaded or not shared_data_components:
        logging.error("Cannot create new chat: Global data not loaded.")
        return None

    chat_id = str(uuid.uuid4()) # Generate unique ID
    logging.info(f"Creating new chat session with ID: {chat_id}")
    agent = TextToSqlAgent(db_path=DB_PATH, agent_id=chat_id)

    # Initialize the agent instance using the globally loaded data
    if agent.initialize_from_loaded_data(
        embed_model=shared_data_components["embed_model"],
        full_schema=shared_data_components["full_schema"],
        detailed_schema_analysis=shared_data_components["detailed_schema_analysis"],
        query_engine=shared_data_components["query_engine"],
    ):
        chat_agents[chat_id] = agent
        logging.info(f"Successfully created and initialized agent for chat {chat_id}")
        return chat_id
    else:
        logging.error(f"Failed to initialize agent for new chat {chat_id}")
        agent.shutdown() # Clean up partially initialized agent
        return None

# --- Initial Cleanup & Potential Load --- 
cleanup_cache_dirs(delete_db=False) # Clean caches on startup, keep existing DB if present
if os.path.exists(DB_PATH):
    logging.info("Existing database found on startup. Attempting to load data globally.")
    if load_global_data(force_regenerate=False):
        # Optionally create a default chat session on startup if data loads
        create_new_chat_session()
    else:
        logging.error("Failed to load data from existing DB on startup.")

# --- Routes --- 
@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/status', methods=['GET'])
def get_status():
    """Returns whether data is loaded and ready for chatting.
    --- 
    tags:
      - Status
    responses:
      200:
        description: Data loading status.
        schema:
          id: StatusResponse
          properties:
            is_data_loaded:
              type: boolean
              description: True if data is loaded and ready to start chats.
            active_chats:
              type: array
              items:
                type: object
                properties:
                  chat_id:
                     type: string
                  title:
                     type: string
              description: List of active chat sessions with their IDs and titles.
    """
    global is_data_loaded, chat_agents
    # Return list of dicts {chat_id: id, title: title}
    chat_list_with_titles = [
        {"chat_id": agent_id, "title": agent.chat_title} 
        for agent_id, agent in chat_agents.items()
    ]
    return jsonify({
        "is_data_loaded": is_data_loaded,
        # "active_chat_ids": list(chat_agents.keys()) # Old way
        "active_chats": chat_list_with_titles # New way
        })

# --- Chat Management Endpoints ---
@app.route('/chats', methods=['GET'])
def get_chats():
    """Gets a list of active chat session IDs.
    --- 
    tags:
      - Chat Management
    responses:
      200:
        description: List of active chat IDs.
        schema:
          type: array
          items:
             type: string
    """
    global chat_agents
    return jsonify(list(chat_agents.keys()))

@app.route('/chats', methods=['POST'])
def create_chat():
    """Creates a new chat session.
    Requires data to be loaded first.
    --- 
    tags:
      - Chat Management
    responses:
      201:
        description: New chat created successfully.
        schema:
          id: NewChatResponse
          properties:
            chat_id:
              type: string
              description: The ID of the newly created chat session.
      400:
        description: Data is not loaded yet.
      500:
        description: Failed to create new chat session.
    """
    global is_data_loaded
    if not is_data_loaded:
        return jsonify({"error": "Data not loaded. Cannot create new chat."}), 400

    new_chat_id = create_new_chat_session()
    if new_chat_id:
        return jsonify({"chat_id": new_chat_id}), 201 # 201 Created
    else:
        return jsonify({"error": "Failed to create new chat session."}), 500

@app.route('/chats/<string:chat_id>', methods=['DELETE'])
def delete_chat(chat_id):
    """Deletes a specific chat session.
    --- 
    tags:
      - Chat Management
    parameters:
      - name: chat_id
        in: path
        type: string
        required: true
        description: The ID of the chat session to delete.
    responses:
      200:
        description: Chat session deleted successfully.
        schema:
          id: DeleteChatSuccess
          properties:
            message:
              type: string
      404:
        description: Chat session not found.
    """
    global chat_agents
    agent_to_delete = chat_agents.pop(chat_id, None)
    if agent_to_delete:
        agent_to_delete.shutdown()
        gc.collect()
        logging.info(f"Deleted chat session: {chat_id}")
        return jsonify({"message": f"Chat session {chat_id} deleted."}), 200
    else:
        logging.warning(f"Attempted to delete non-existent chat ID: {chat_id}")
        return jsonify({"error": f"Chat session {chat_id} not found."}), 404

# --- Query Endpoint ---
@app.route('/query', methods=['POST'])
def handle_query():
    """Handles user queries for a specific chat session.
    Receives a question, chat ID, and optionally a model name.
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
            - chat_id
          properties:
            question:
              type: string
              description: The user's natural language question.
            chat_id:
              type: string
              description: The ID of the chat session this query belongs to.
            model:
              type: string
              description: (Optional) LLM model name. Defaults to server default.
            generate_title:
                type: boolean
                description: (Optional) Set to true if this is the first message and a title should be generated.
    responses:
      200:
        description: Agent's response.
        schema:
          id: QueryResponse # Assuming QueryResponse schema exists from previous def
          properties:
            response_type:
              type: string
            message:
              type: string
            sql_query:
              type: string
              nullable: true
            chat_title:
               type: string
               description: The current (potentially updated) title of the chat.
      400:
        description: Bad request (e.g., missing chat_id, missing question, data not loaded).
      404:
        description: Chat session not found.
      500:
        description: Internal server error.
    """
    global is_data_loaded, chat_agents

    if not is_data_loaded:
         return jsonify({"error": "Data not loaded. Please load data first."}), 400

    try:
        data = request.get_json()
        if data is None:
            logging.error("Received query request with invalid/empty JSON body.")
            return jsonify({"error": "Invalid request body."}), 400
            
        user_question = data.get('question')
        chat_id = data.get('chat_id')
        model_name = data.get('model', DEFAULT_LLM_MODEL)
        should_generate_title = data.get('generate_title', False) # Get the flag

        if not chat_id:
            return jsonify({"error": "Missing chat_id in request."}), 400
        if not user_question:
            return jsonify({"error": "No question provided."}), 400

        # Get the agent instance for this chat
        agent = chat_agents.get(chat_id)
        if not agent:
            logging.warning(f"Query received for non-existent chat ID: {chat_id}")
            return jsonify({"error": f"Chat session {chat_id} not found."}), 404
        if not agent.is_initialized:
            logging.error(f"Query received for uninitialized agent (Chat ID: {chat_id})")
            return jsonify({"error": f"Agent for chat {chat_id} is not ready."}), 500 # Internal error state

        logging.info(f"Received query for chat '{chat_id}' using model '{model_name}': '{user_question}'")

        # --- Title Generation (if requested and needed) ---
        logging.debug(f"Checking title generation for chat {chat_id}: should_generate={should_generate_title}, current_agent_title='{agent.chat_title}'") # DEBUG LOG
        if should_generate_title and agent.chat_title == "New Chat":
            logging.info(f"Request to generate title for chat {chat_id} based on: '{user_question}'")
            # Get LLM instance (might be needed for title gen)
            # Reuse the LLM intended for the main query for efficiency
            llm_instance_for_title = agent._get_llm(model_name)
            if llm_instance_for_title:
                new_title = generate_chat_title(user_question, llm_instance_for_title)
                if new_title:
                    agent.chat_title = new_title # Update agent's title
                    logging.info(f"Successfully updated title for chat {chat_id} to: '{new_title}'")
                else:
                    logging.warning(f"Failed to generate title for chat {chat_id}. Keeping default.")
            else:
                logging.warning(f"Could not get LLM instance to generate title for chat {chat_id}.")
        # --- End Title Generation ---

        response = agent.process_query(user_question, model_name)

        logging.info(f"Sending response for chat {chat_id}: Type={response.get('type')}")

        return jsonify({
            "response_type": response.get("type"),
            "message": response.get("message"),
            "sql_query": response.get("sql_query"),
            "chat_title": agent.chat_title # Always return the current title
        })

    except Exception as e:
        logging.error(f"Error handling /query request: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred."}), 500

# --- Data Management Endpoints ---
@app.route('/upload-data', methods=['POST'])
def upload_data():
    """Handles CSV uploads, converts to SQLite, REPLACES ALL existing data/chats,
    loads data globally, and starts ONE new chat session.
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
        description: One or more CSV files.
    responses:
      201:
        description: Files uploaded, data loaded, and initial chat created.
        schema:
          id: UploadSuccessResponse
          properties:
            message:
              type: string
            new_chat_id:
              type: string
              description: The ID of the initial chat session created with the new data.
      400:
        description: Bad request (e.g., no files).
      500:
        description: Internal server error during processing.
    """
    global is_data_loaded, shared_data_components, chat_agents

    if 'files' not in request.files:
        return jsonify({"error": "No file part"}), 400
    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        return jsonify({"error": "No selected files"}), 400

    # 1. Delete ALL existing data, caches, and shutdown agents
    logging.warning("Upload request received. Deleting ALL existing data and caches...")
    del_errors, del_warnings = cleanup_cache_dirs(delete_db=True)
    if del_errors:
         logging.error(f"Errors occurred during pre-upload cleanup: {del_errors}. Aborting upload.")
         # Return 500 if cleanup failed critically
         return jsonify({"error": f"Failed to clean up existing data before upload: {del_errors}", "warnings": del_warnings}), 500

    # State should be reset by cleanup_cache_dirs(delete_db=True)
    # is_data_loaded = False
    # shared_data_components = None
    # chat_agents = {}

    # 2. Save uploaded files temporarily
    saved_files = []
    # Ensure upload folder exists (might have been deleted)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    for file in files:
        if file and file.filename.lower().endswith('.csv'):
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            try:
                file.save(filepath)
                saved_files.append(filepath)
                logging.info(f"Saved uploaded file: {filepath}")
            except Exception as e:
                 logging.error(f"Error saving uploaded file {file.filename}: {e}")
                 # Clean up already saved files before erroring?
                 return jsonify({"error": f"Failed to save uploaded file {file.filename}"}), 500
        else:
            logging.warning(f"Skipping non-CSV file: {file.filename}")

    if not saved_files:
        return jsonify({"error": "No valid CSV files were uploaded or saved."}), 400

    # 3. Convert CSVs to SQLite DB
    try:
        conn = sqlite3.connect(DB_PATH)
        logging.info(f"Creating/Replacing database: {DB_PATH}")
        for csv_path in saved_files:
            table_name = os.path.splitext(os.path.basename(csv_path))[0]
            table_name = "".join(c if c.isalnum() else '_' for c in table_name).strip('_')
            if not table_name: table_name = f"data_{uuid.uuid4().hex[:6]}" # Ensure unique fallback
            logging.info(f"Processing {csv_path} into table '{table_name}'")

            df = None
            encodings_to_try = ['utf-8', 'cp1252', 'latin1']
            for encoding in encodings_to_try:
                try:
                    df = pd.read_csv(csv_path, encoding=encoding)
                    logging.info(f"Read {csv_path} using encoding '{encoding}'")
                    break
                except UnicodeDecodeError:
                    logging.warning(f"Failed read {csv_path} encoding '{encoding}'. Trying next...")
                except Exception as e:
                    logging.error(f"Error reading {csv_path} with '{encoding}': {e}")
                    df = None
                    break

            if df is None:
                logging.error(f"Failed read {csv_path}. Skipping.")
                # Clean up DB if one file fails? Or allow partial load?
                # Let's continue for now, but report issue later if load fails.
                continue

            df.columns = ["".join(c if c.isalnum() else '_' for c in str(col)).strip('_') or f'col_{i}' for i, col in enumerate(df.columns)]
            # Ensure no empty column names
            df.columns = [f'col_{i}' if not name else name for i, name in enumerate(df.columns)]

            df.to_sql(table_name, conn, if_exists='replace', index=False)
            logging.info(f"Loaded data into table '{table_name}'")
        conn.close()
    except Exception as e:
        logging.error(f"Error converting CSV to SQLite: {e}", exc_info=True)
        cleanup_cache_dirs(delete_db=True) # Attempt cleanup on failure
        return jsonify({"error": f"Error processing CSV files: {e}"}), 500

    # 4. Trigger global data loading/indexing
    if not load_global_data(force_regenerate=True): # Force regenerate on new upload
        cleanup_cache_dirs(delete_db=True) # Attempt cleanup on failure
        return jsonify({"error": "Data loaded to DB, but global indexing failed. Check logs."}), 500

    # 5. Create the *first* chat session for the newly loaded data
    initial_chat_id = create_new_chat_session()
    if not initial_chat_id:
        cleanup_cache_dirs(delete_db=True) # Attempt cleanup on failure
        return jsonify({"error": "Data loaded and indexed, but failed to create initial chat session."}), 500

    return jsonify({
        "message": f"Uploaded {len(saved_files)} CSV(s), loaded data, and created initial chat.",
        "new_chat_id": initial_chat_id
        }), 201 # 201 Created

@app.route('/remove-data', methods=['POST'])
def remove_data():
    """Removes ALL data: the database, caches, and ALL chat sessions.
    --- 
    tags:
      - Data Management
    responses:
      200:
        description: All data and chats removed.
        schema:
          id: RemoveSuccess
          properties:
            message:
              type: string
            warnings:
              type: array
              items:
                type: string
      500:
        description: Critical errors during cleanup.
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
    logging.warning("Received request to remove ALL data.")
    # cleanup_cache_dirs handles agent shutdown and state reset
    errors, warnings = cleanup_cache_dirs(delete_db=True)

    if errors:
         # If critical errors occurred (DB deletion failed etc.)
         return jsonify({"message": f"Attempted to remove data. Critical errors encountered: {errors}", "errors": errors, "warnings": warnings}), 500
    else:
        success_message = "Successfully removed all data (DB, caches) and chat sessions." 
        if warnings:
            success_message += " Note: Some non-critical cleanup warnings occurred (e.g., EBUSY)."
        return jsonify({"message": success_message, "warnings": warnings}), 200

# --- Removed /refresh-index endpoint --- 
# Refreshing is now implicitly handled by uploading new data (which forces regeneration)
# or potentially by restarting the server if needed for cache issues.
# Keeping the endpoint would require careful state management.

# --- Application Runner ---
if __name__ == '__main__':
    # Use 0.0.0.0 to make it accessible outside the container
    app.run(host='0.0.0.0', port=5000, debug=False)

# --- Application Runner ---
if __name__ == '__main__':
    # Use 0.0.0.0 to make it accessible outside the container
    app.run(host='0.0.0.0', port=5000, debug=False) # Turn debug off for production 