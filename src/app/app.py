import os
import sqlite3
import json
import shutil  # For deleting directories
import concurrent.futures
import time # Import time module
from typing import List, Tuple, Dict, Optional, Union
import logging
from llama_index.core import Document # Still needed for type hints
from llama_index.core import Settings # Still needed? Check usage
from llama_index.core import PromptTemplate # Still needed? Check usage - Yes, for LLM interface
from llama_index.core.base.base_query_engine import BaseQueryEngine # For type hint
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.llms.gemini import Gemini

# Import the new data handler module
import data_handler 
# Import the new LLM interface module
import llm_interface
from vector_store import VectorStoreManager # Import the new manager

# ------------------------------
# Configure Logging
# ------------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# ------------------------------
# Load API Key from secrets
# ------------------------------
try:
    with open("secrets/openai_api_key.txt", "r") as key_file:
        os.environ["OPENAI_API_KEY"] = key_file.read().strip()
except FileNotFoundError:
    logging.error(
        "API key file 'secrets/openai_api_key.txt' not found. Please create it."
    )
    # Depending on the desired behavior, you might want to exit or raise an exception here.
    # For now, we'll proceed, but OpenAI calls will fail.
    pass

# Disable tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ------------------------------
# Conversation History Class
# ------------------------------
class ConversationHistory:
    def __init__(self, max_history: int = 5):
        self.history: List[Dict] = []
        self.max_history = max_history
        logging.info(f"ConversationHistory initialized with max_history={max_history}")

    def add_interaction(
        self,
        question: str,
        response_type: str,
        sql_query: Optional[str] = None,
        results: Optional[str] = None,
        analysis: Optional[str] = None,
    ):
        interaction = {
            "question": question,
            "response_type": response_type,  # e.g., "sql_analysis", "direct_answer", "clarification_needed"
            "sql_query": sql_query,
            "results": results,
            "analysis": analysis,
        }
        self.history.append(interaction)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        logging.debug(
            f"Added interaction to history. Current length: {len(self.history)}"
        )

    def get_formatted_history(self) -> str:
        if not self.history:
            return ""

        formatted = "Previous conversation context:\n"
        for i, interaction in enumerate(self.history, 1):
            formatted += f"Interaction {i}:\n"
            formatted += f"  User Question: {interaction['question']}\n"
            if interaction["sql_query"]:
                formatted += f"  SQL Generated: {interaction['sql_query']}\n"
            if interaction["results"]:
                formatted += f"  Query Results: {interaction['results']}\n"
            if interaction["analysis"]:
                formatted += f"  Response/Analysis: {interaction['analysis']}\n"
            formatted += "-" * 40 + "\n"
        return formatted


# ------------------------------
# TextToSqlAgent Class
# ------------------------------
class TextToSqlAgent:
    def __init__(
        self,
        db_path: str,
        cache_file: str = "schema_cache.json",
        default_model: str = "gpt-4-turbo",
    ):
        logging.info("Initializing TextToSqlAgent object...")
        # Configuration details stored, but initialization deferred
        persist_dir = "vector_store_cache"  # Directory to store the index
        self.db_path = db_path
        self.cache_file = cache_file
        self.default_model = (
            default_model  # Model for schema analysis and default queries
        )
        self.initialized_llms: Dict[
            str, Union[OpenAI, Gemini]
        ] = {}  # Cache for dynamically loaded LLMs
        self.embed_model = None  # Initialized in load_and_index
        # self.index = None  # Managed by VectorStoreManager
        self.query_engine: Optional[BaseQueryEngine] = None  # Set by load_and_index
        self.conn = None  # Initialized in load_and_index
        self.cursor = None  # Initialized in load_and_index
        self.full_schema = ""  # Initialized in load_and_index
        self.conversation_history = ConversationHistory()
        # Initialize the Vector Store Manager
        self.vector_store_manager = VectorStoreManager(persist_dir=persist_dir)
        self.is_initialized = False  # Flag to track if indexing is complete
        # Defer actual initialization until load_and_index is called

    def load_and_index(self):
        """Loads data, initializes components, builds/loads index. Called after DB exists."""
        if self.is_initialized:
            logging.info("Agent already initialized.")
            return True

        if not os.path.exists(self.db_path):
            logging.error(
                f"Database path does not exist: {self.db_path}. Cannot initialize."
            )
            return False

        logging.info(f"Starting initialization and indexing for DB: {self.db_path}")
        
        # --- Initialize Embeddings --- 
        logging.info("Initializing Embeddings...")
        try:
            # Initialize Embeddings first
            logging.info("Initializing OpenAI Embeddings (text-embedding-3-small)...")
            self.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
            if not self.embed_model:
                 raise ValueError("Embedding model initialization failed.")
        except Exception as e:
            logging.error(f"Error initializing Embeddings: {e}", exc_info=True)
            return False # Cannot proceed without embeddings

        # --- Initialize dedicated LLM for Schema Analysis --- 
        # (Moved initialization here to pass to data_handler)
        schema_analysis_model_name = "gpt-4-turbo" # Or use 3.5-turbo if preferred
        schema_analysis_llm = None
        try:
            logging.info(f"Initializing {schema_analysis_model_name} specifically for schema analysis...")
            schema_analysis_llm = OpenAI(model=schema_analysis_model_name, temperature=0.01)
            if not os.environ.get("OPENAI_API_KEY"):
                 raise ValueError("OpenAI API Key not found, needed for schema analysis.")
        except Exception as e:
            logging.error(f"Failed to initialize {schema_analysis_model_name} for schema analysis: {e}. Schema summaries might be basic.", exc_info=True)
            # Proceed even if LLM fails, data_handler generates basic summaries

        # --- Load Schema using Data Handler --- 
        logging.info("Loading DB schema and column summaries using data_handler...")
        try:
            self.full_schema, schema_docs = data_handler.load_db_schema_and_summaries(
                self.db_path, self.cache_file, schema_analysis_llm
            )
            if not self.full_schema or not schema_docs:
                 logging.error("Failed to load schema or generate documents from data_handler.")
                 return False
            logging.info("Database schema loaded/analyzed.")
            logging.debug(f"Full Schema Snippet:\n{self.full_schema[:500]}...")
        except Exception as e:
             logging.error(f"Error calling data_handler.load_db_schema_and_summaries: {e}", exc_info=True)
             return False

        # --- Establish Persistent DB Connection for Agent --- 
        logging.info("Establishing persistent DB connection for agent...")
        db_conn_info = data_handler.get_db_connection(self.db_path)
        if db_conn_info:
            self.conn, self.cursor = db_conn_info
        else:
            logging.error(f"Failed to establish persistent DB connection to {self.db_path}. Cannot execute queries.")
            return False

        # --- Initialize Vector Store Index using Manager --- 
        logging.info("Initializing Vector Store Index via VectorStoreManager...")
        try:
             # The manager handles checking cache, loading, or building
             _, self.query_engine = self.vector_store_manager.load_or_build_index(
                 schema_docs=schema_docs, 
                 embed_model=self.embed_model
             )
             # Check if the manager succeeded
             if not self.query_engine:
                  raise RuntimeError("VectorStoreManager failed to load or build index/query engine.")
             logging.info("VectorStoreManager successfully provided query engine.")

        except Exception as e:
             # Catch errors from the manager call or the RuntimeError above
             logging.error(f"Error during vector store initialization via manager: {e}", exc_info=True)
             if self.conn:
                  self.conn.close() # Cleanup connection on index failure
             return False
        # --- Vector Store Initialization Complete --- 

        self.is_initialized = True
        logging.info("Agent initialization and indexing complete.")
        return True

    def _validate_question(self, user_question: str, llm: llm_interface.LlmType) -> Dict[str, str]:
        """
        Uses LLM (via llm_interface) to validate if the question needs SQL,
        can be answered directly, or requires clarification.
        """
        logging.info(f"Validating question: '{user_question}'")
        history_context = self.conversation_history.get_formatted_history()

        # Call the validation function from the llm_interface module
        return llm_interface.validate_question(
            question=user_question,
            schema=self.full_schema,
            history=history_context,
            llm=llm
        )
            
    def _generate_sql_query(self, user_question: str, llm: llm_interface.LlmType) -> Tuple[Optional[str], str]:
        """Generates SQL query using context, schema, history via llm_interface."""
        logging.info(f"Generating SQL for question: '{user_question}'")
        # Ensure query engine is available before attempting context retrieval
        if not self.query_engine:
             logging.error("Query engine not available for context retrieval.")
             return None, "Error: Query engine not initialized."
             
        try:
            # Retrieve context using the agent's query engine
            retrieved_context = str(self.query_engine.query(user_question))
            logging.debug(f"Retrieved context for SQL generation: {retrieved_context}")
        except Exception as e:
            logging.error(f"Error retrieving context from query engine: {e}", exc_info=True)
            retrieved_context = "Context retrieval failed." # Provide fallback context

        history_context = self.conversation_history.get_formatted_history()

        # Call the SQL generation function from the llm_interface module
        return llm_interface.generate_sql(
            question=user_question,
            schema=self.full_schema,
            history=history_context,
            context=retrieved_context,
            llm=llm
        )
            
    def _execute_sql_query(self, sql_query: str) -> Union[List[Tuple], str]:
        """Executes the generated SQL query against the database."""
        logging.info(f"Executing SQL: {sql_query}")
        if not self.cursor:
            logging.error("Database cursor is not initialized.")
            return "Error: Database connection is not available."
        try:
            self.cursor.execute(sql_query)
            results = self.cursor.fetchall()
            logging.info(f"Query executed successfully, {len(results)} rows returned.")
            return results
        except sqlite3.Error as e:
            logging.error(f"Error executing SQL query: {e}", exc_info=True)
            return f"Error executing query: {e}"
        except Exception as e:  # Catch other potential errors
            logging.error(
                f"An unexpected error occurred during query execution: {e}",
                exc_info=True,
            )
            return f"An unexpected error occurred: {e}"
            
    def _generate_analysis(self, user_question: str, sql_query: str, query_results: Union[List[Tuple], str], llm: llm_interface.LlmType) -> str:
        """Generates a natural language analysis of the query results via llm_interface."""
        logging.info("Generating analysis for query results.")
        history_context = self.conversation_history.get_formatted_history()

        # Call the analysis generation function from the llm_interface module
        # Pass the cursor for header extraction
        return llm_interface.generate_analysis(
             question=user_question,
             sql=sql_query,
             query_results=query_results,
             history=history_context,
             llm=llm,
             cursor=self.cursor # Pass the agent's cursor
        )
            
    def process_query(self, user_question: str, model_name: str) -> Dict[str, Optional[str]]:
        """
        Processes the user's question: validates, potentially clarifies,
        generates/executes SQL, and returns analysis or response. Requires agent to be initialized.
        """
        # Check for initialization, including the query engine which is set by the manager
        if not self.is_initialized or not self.query_engine or not self.conn:
            logging.error("Agent not initialized (or query engine missing). Cannot process query.")
            return {
                "type": "error",
                "message": "Agent is not ready. Please ensure data is loaded and indexed.",
                "sql_query": None,
            }

        logging.info(f"Processing user query: '{user_question}'")

        # Get the appropriate LLM instance for this request
        try:
            llm = self._get_llm(model_name)
            if not llm:
                raise ValueError(f"Could not initialize LLM for model: {model_name}")
        except Exception as e:
            logging.error(f"Failed to get LLM instance: {e}", exc_info=True)
            return {
                "type": "error",
                "message": f"Error initializing language model: {e}",
                "sql_query": None,
            }

        # --- Call agent methods that now use llm_interface --- 
        validation_result = self._validate_question(user_question, llm)
        action = validation_result["action"]
        details = validation_result["details"]

        response = {"type": action, "message": None, "sql_query": None}

        if action == "DIRECT_ANSWER":
            logging.info("Action: DIRECT_ANSWER")
            response["message"] = details
            self.conversation_history.add_interaction(
                question=user_question, response_type="direct_answer", analysis=details
            )

        elif action == "CLARIFICATION_NEEDED":
            logging.info("Action: CLARIFICATION_NEEDED")
            response["message"] = details
            self.conversation_history.add_interaction(
                question=user_question,
                response_type="clarification_needed",
                analysis=details,  # Store the clarification question asked
            )

        elif action == "SQL_NEEDED":
            logging.info("Action: SQL_NEEDED")
            sql_query, raw_llm_sql_response = self._generate_sql_query(
                user_question, llm
            )
            response["sql_query"] = sql_query

            if sql_query:
                query_results = self._execute_sql_query(sql_query)
                analysis = self._generate_analysis(
                    user_question, sql_query, query_results, llm
                )
                response["message"] = analysis
                self.conversation_history.add_interaction(
                    question=user_question,
                    response_type="sql_analysis",
                    sql_query=sql_query,
                    results=str(query_results)
                    if not isinstance(query_results, str)
                    else query_results,
                    analysis=analysis,
                )
            else:
                logging.error("SQL generation failed.")
                error_msg = raw_llm_sql_response if raw_llm_sql_response else "I encountered an error trying to generate the SQL query needed to answer your question."
                response["message"] = error_msg
                self.conversation_history.add_interaction(
                    question=user_question,
                    response_type="error",
                    analysis=response["message"],
                )

        else:  # Should not happen based on validation logic
            logging.error(f"Invalid action '{action}' received from validation.")
            response["type"] = "error"
            response["message"] = "An unexpected internal error occurred during question validation."
            self.conversation_history.add_interaction(
                question=user_question,
                response_type="error",
                analysis=response["message"],
            )

        logging.info(f"Finished processing query. Response type: {response['type']}")
        return response
        
    def _get_llm(self, model_name: str) -> Optional[llm_interface.LlmType]:
        """Gets an initialized LLM instance, caching it if necessary."""
        if model_name in self.initialized_llms:
            logging.debug(f"Using cached LLM instance for: {model_name}")
            return self.initialized_llms[model_name]

        logging.info(f"Initializing LLM for: {model_name}")
        llm_instance: Optional[llm_interface.LlmType] = None # Use type alias
        try:
            if model_name.startswith("gpt-"):
                llm_instance = OpenAI(
                    model=model_name,
                    temperature=0.01,
                    max_new_tokens=500,  # Consider adjusting based on model
                    request_timeout=60,
                )
            elif (
                model_name.startswith("models/gemini-")
                or model_name == "gemini-2.5-pro-exp-03-25"
            ):
                if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
                    logging.error(
                        "GOOGLE_APPLICATION_CREDENTIALS environment variable not set. Cannot initialize Gemini via ADC."
                    )
                    return None
                gemini_model_id = model_name
                if not gemini_model_id.startswith(
                    "models/"
                ) and not gemini_model_id.startswith("tunedModels/"):
                    gemini_model_id = f"models/{gemini_model_id}"
                    logging.info(
                        f"Prepended 'models/' prefix. Using: {gemini_model_id}"
                    )
                llm_instance = Gemini(
                    model_name=gemini_model_id,
                    temperature=0.01,
                )
            else:
                logging.error(f"Unsupported model name provided: {model_name}")
                return None

            self.initialized_llms[model_name] = llm_instance
            logging.info(f"Successfully initialized LLM for: {model_name}")
            return llm_instance

        except Exception as e:
            logging.error(f"Error initializing LLM {model_name}: {e}", exc_info=True)
            return None  # Failed to initialize
            
    def _print_vector_store_samples(self, num_samples=3):
        """Prints sample documents from the vector store."""
        # This might need adjustment if self.index is no longer directly accessible
        # or if manager should provide this functionality.
        # For now, let's assume self.index might be None if manager failed, 
        # or we need to get the index from the manager if we didn't store it.
        # Let's try getting it from the manager if we didn't store self.index
        # Assuming self.query_engine exists implies self.index exists inside manager or was loaded.
        # Let's refine this: We should probably store self.index from the manager's return value.
        # EDIT: Ok, I'll add storing self.index back.
        
        # We need self.index to access docstore. Let's get it from query_engine.
        if not self.query_engine or not hasattr(self.query_engine, 'retriever') or not hasattr(self.query_engine.retriever, '_index'):
             # This access pattern might change with LlamaIndex versions
             # A safer approach might be storing self.index alongside self.query_engine
             logging.warning("Cannot access index/docstore via query engine for printing samples.")
             return
             
        index = self.query_engine.retriever._index # Accessing internal attribute, potentially fragile

        if not index or not index.docstore:
            logging.warning(
                "Vector store index/docstore not initialized or empty. Cannot print samples."
            )
            return

        logging.debug("\n" + "=" * 80)
        logging.debug("VECTOR STORE SAMPLES:")
        logging.debug("=" * 80)
        try:
            documents = list(index.docstore.docs.values())
            if not documents:
                logging.debug("No documents found in the vector store.")
                return

            for i, doc in enumerate(documents[:num_samples]):
                logging.debug(f"\nSample Document {i+1}:")
                logging.debug("-" * 40)
                logging.debug(doc.text)
                logging.debug("-" * 40)

            logging.debug("\n" + "=" * 80 + "\n")
        except Exception as e:
            logging.error(f"Error accessing vector store documents: {e}", exc_info=True)
            
    def shutdown(self):
        """Closes DB connection and resets state."""
        logging.info("Shutting down agent resources...")
        if self.conn:
            self.conn.close()
            self.conn = None
        self.cursor = None
        # self.index = None # Now managed by vector_store_manager
        self.query_engine = None
        self.initialized_llms = {}
        self.embed_model = None
        self.full_schema = ""
        # self.vector_store_manager = None # Reset manager? Maybe not necessary if stateless
        self.is_initialized = False
        logging.info("Agent resources released.")


# Note: The main execution block (`if __name__ == "__main__":`)
# has been removed as this script will now be imported and used by the server.
# You would typically instantiate and use the TextToSqlAgent from your Flask app.
