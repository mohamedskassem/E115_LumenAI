import logging
from flask import Flask, request, jsonify, render_template
from app import TextToSqlAgent # Assuming app.py is in the same directory

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# --- Agent Initialization ---
# Load environment variables or config for DB path
DB_PATH = "./output/adventure_works.db" # Make sure this path is accessible inside the container
CACHE_FILE = "schema_cache.json"
DEFAULT_MODEL = "gpt-4-turbo" # Define default model here or load from config

try:
    logging.info("Initializing TextToSqlAgent...")
    agent = TextToSqlAgent(db_path=DB_PATH, cache_file=CACHE_FILE, default_model=DEFAULT_MODEL)
    logging.info("TextToSqlAgent initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize TextToSqlAgent: {e}", exc_info=True)
    # Optionally, you could prevent the Flask app from starting
    # raise SystemExit("Agent initialization failed.")
    agent = None # Set agent to None if initialization fails

# --- Routes ---
@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def handle_query():
    """Handles user queries sent from the frontend."""
    if agent is None:
        return jsonify({"error": "Agent is not initialized. Check server logs."}), 500

    try:
        data = request.get_json()
        user_question = data.get('question')
        # Get model name from request, fallback to default if not provided
        model_name = data.get('model', DEFAULT_MODEL)

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

# --- Application Runner ---
if __name__ == '__main__':
    # Use 0.0.0.0 to make it accessible outside the container
    app.run(host='0.0.0.0', port=5000, debug=False) # Turn debug off for production 