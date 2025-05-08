import os
import sqlite3
import json

from typing import List, Tuple, Dict, Optional, Union
import logging
import google.generativeai as genai  # NEW IMPORT

from llama_index.core.base.base_query_engine import BaseQueryEngine  # For type hint
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# Import the new data handler module
import data_handler

# Import the new LLM interface module
import llm_interface
from vector_store import VectorStoreManager  # Import the new manager


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


try:
    with open("secrets/google_api_key.txt", "r") as key_file:
        os.environ["GOOGLE_API_KEY"] = key_file.read().strip()
        # Configure google.generativeai globally if key found
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        logging.info(
            "Configured google-generativeai using GOOGLE_API_KEY from secrets."
        )
except FileNotFoundError:
    logging.info(
        "secrets/google_api_key.txt not found. Will rely on Application Default Credentials (ADC) "
        "if GOOGLE_APPLICATION_CREDENTIALS is set."
    )
    # No explicit configure call here, genai library handles ADC lookup
    pass  # Proceed, Gemini calls might use ADC

# Disable tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ------------------------------
# Conversation History Class
# ------------------------------
class ConversationHistory:
    def __init__(self, max_history: int = 50):
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
        db_path: str,  # Keep db_path for reference, maybe connection management
        agent_id: str = "default_agent",  # Add an identifier
        chat_title: str = "New Chat",  # Initialize title
    ):
        """Initializes agent configuration, deferring resource loading."""
        self.agent_id = agent_id
        logging.info(f"Initializing TextToSqlAgent object (ID: {self.agent_id})...")
        self.db_path = db_path
        self.initialized_llms: Dict[str, Union[OpenAI, genai.GenerativeModel]] = (
            {}
        )  # Cache for dynamically loaded LLMs per agent

        # These will be set by `initialize_from_loaded_data`
        self.embed_model: Optional[OpenAIEmbedding] = None
        self.query_engine: Optional[BaseQueryEngine] = None
        self.conn: Optional[sqlite3.Connection] = None
        self.cursor: Optional[sqlite3.Cursor] = None
        self.full_schema: str = ""
        self.detailed_schema_analysis: Optional[Dict] = None
        self.is_initialized: bool = False
        self.chat_title: str = chat_title  # Initialize title

        self.conversation_history = (
            ConversationHistory()
        )  # Each agent gets its own history
        logging.info(f"Agent {self.agent_id} created with fresh conversation history.")

    # --- Data Loading and Indexing (Static/Class Method - Run Once Globally) ---
    @staticmethod
    def load_and_index_data(
        db_path: str,
        persist_dir: str = "vector_store_cache",
        force_regenerate_analysis: bool = False,
    ) -> Optional[Dict[str, object]]:
        """
        Loads data, initializes shared components (embeddings, schema, index),
        and returns them for agents to use. This should be run once per dataset load.

        Args:
            db_path: Path to the database.
            persist_dir: Directory for vector store cache.
            force_regenerate_analysis: Force regeneration of schema analysis and index.

        Returns:
            A dictionary containing initialized components ('embed_model', 'full_schema',
            'detailed_schema_analysis', 'query_engine', 'db_connection') or None on failure.
        """
        logging.info(
            f"Starting GLOBAL data loading/indexing for DB: {db_path} (Force Regen: {force_regenerate_analysis})"
        )

        if not os.path.exists(db_path):
            logging.error(
                f"Database path does not exist: {db_path}. Cannot initialize."
            )
            return None

        # --- Initialize Embeddings (Shared) ---
        embed_model = None
        try:
            logging.info(
                "Initializing SHARED OpenAI Embeddings (text-embedding-3-small)..."
            )
            # Explicitly pass API key
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY environment variable not found.")
            embed_model = OpenAIEmbedding(
                model="text-embedding-3-small",
                api_key=openai_api_key
            )
            if not embed_model:
                raise ValueError("Shared Embedding model initialization failed.")
        except Exception as e:
            logging.error(f"Error initializing SHARED Embeddings: {e}", exc_info=True)
            return None  # Cannot proceed without embeddings

        # --- Initialize dedicated LLM for Schema Analysis (Temporary for this step) ---
        schema_analysis_model_name = "gpt-4-turbo"  # Or choose another robust model
        schema_analysis_llm = None
        try:
            logging.info(
                f"Initializing {schema_analysis_model_name} temporarily for schema analysis..."
            )
            # Use a temporary instance, don't cache it in the static method
            if schema_analysis_model_name.startswith("gpt-"):
                # Explicitly pass API key if found
                openai_api_key = os.environ.get("OPENAI_API_KEY")
                if openai_api_key:
                    schema_analysis_llm = OpenAI(
                        model=schema_analysis_model_name,
                        api_key=openai_api_key
                    )
                else:
                    logging.warning(
                        "OPENAI_API_KEY environment variable not found, needed for schema analysis. "
                        "Analysis will be basic."
                    )
            elif schema_analysis_model_name.startswith("gemini-"):
                # Check if genai is configured (via key or ADC)
                if os.getenv("GOOGLE_API_KEY") or os.getenv(
                    "GOOGLE_APPLICATION_CREDENTIALS"
                ):
                    try:
                        schema_analysis_llm = genai.GenerativeModel(
                            schema_analysis_model_name
                        )
                    except Exception as gemini_e:
                        logging.error(
                            "Failed to initialize temporary Gemini for schema analysis: "
                            f"{gemini_e}"
                        )
                else:
                    logging.warning(
                        "Google API Key/Credentials not found, needed for schema analysis with Gemini. "
                        "Analysis will be basic."
                    )
            else:
                logging.warning(
                    f"Unsupported model for schema analysis: {schema_analysis_model_name}"
                )

        except Exception as e:
            logging.error(
                f"Failed to initialize {schema_analysis_model_name} for schema analysis: {e}. "
                f"Schema summaries might be basic.",
                exc_info=True,
            )

        # --- Load Schema and Analysis using Data Handler ---
        full_schema, schema_docs, detailed_schema_analysis = None, None, None
        try:
            logging.info("Loading DB schema and analysis using data_handler...")
            full_schema, schema_docs, detailed_schema_analysis = (
                data_handler.load_db_schema_and_analysis(
                    db_path,
                    schema_analysis_llm,
                    force_regenerate=force_regenerate_analysis,
                )
            )
            if not full_schema or schema_docs is None:
                logging.error(
                    "Failed to load schema or generate documents from data_handler."
                )
                return None
            logging.info("Database schema loaded/analyzed.")
            logging.debug(f"Full Schema Snippet:\n{full_schema[:500]}...")
        except Exception as e:
            logging.error(
                f"Error calling data_handler.load_db_schema_and_analysis: {e}",
                exc_info=True,
            )
            return None

        # --- Initialize Vector Store Index using Manager (Shared Query Engine) ---
        vector_store_manager = VectorStoreManager(persist_dir=persist_dir)
        query_engine = None
        try:
            logging.info(
                "Initializing/Updating SHARED Vector Store Index via VectorStoreManager..."
            )
            # The manager handles checking cache, loading, or building
            _, query_engine = vector_store_manager.load_or_build_index(
                schema_docs=schema_docs,
                embed_model=embed_model,  # Pass the shared embed model
                force_rebuild=force_regenerate_analysis,
            )
            if not query_engine:
                raise RuntimeError(
                    "VectorStoreManager failed to load or build index/query engine."
                )
            logging.info(
                "VectorStoreManager successfully provided SHARED query engine."
            )
        except Exception as e:
            logging.error(
                f"Error during SHARED vector store initialization via manager: {e}",
                exc_info=True,
            )
            return None

        # --- Establish DB Connection (Maybe per-agent or shared carefully?) ---
        # For simplicity in Flask, creating a connection per agent might be safer
        # due to thread-safety concerns with shared SQLite connections across requests.
        # Let's return None here and let the agent instance create its own.
        # db_connection_info = data_handler.get_db_connection(db_path)
        # if not db_connection_info:
        #     logging.error(f"Failed to establish DB connection to {db_path}.")
        #     # Decide if this is fatal. Maybe agents can try connecting individually?
        #     # Let's make it non-fatal for now.
        #     # db_connection_info = None # Set to None if failed

        logging.info("GLOBAL data loading/indexing complete.")
        return {
            "embed_model": embed_model,
            "full_schema": full_schema,
            "detailed_schema_analysis": detailed_schema_analysis,
            "query_engine": query_engine,
            # "db_connection": db_connection_info # Don't return connection
        }

    # --- Instance Initialization ---
    def initialize_from_loaded_data(
        self,
        embed_model: OpenAIEmbedding,
        full_schema: str,
        detailed_schema_analysis: Optional[Dict],
        query_engine: BaseQueryEngine,
        # db_connection_info: Optional[Tuple[sqlite3.Connection, sqlite3.Cursor]]
    ) -> bool:
        """Initializes this agent instance with pre-loaded shared data."""
        logging.info(
            f"Initializing agent instance {self.agent_id} from pre-loaded data..."
        )
        self.embed_model = embed_model
        self.full_schema = full_schema
        self.detailed_schema_analysis = detailed_schema_analysis
        self.query_engine = query_engine

        # Establish a dedicated DB connection for this agent instance
        logging.info(
            f"Establishing dedicated DB connection for agent {self.agent_id}..."
        )
        db_connection_info = data_handler.get_db_connection(self.db_path)
        if db_connection_info:
            self.conn, self.cursor = db_connection_info
            logging.info(
                f"Dedicated DB connection established for agent {self.agent_id}."
            )
        else:
            logging.error(
                f"Failed to establish DB connection for agent {self.agent_id}. Queries will fail."
            )
            self.is_initialized = False
            return False  # Explicitly return False here

        # This part should only run if the connection was successful
        self.is_initialized = True
        logging.info(f"Agent instance {self.agent_id} initialized successfully.")
        return True

    # --- Core Agent Logic (Methods operate on instance data) ---

    def _validate_question(
        self, user_question: str, llm: llm_interface.LlmType
    ) -> Dict[str, str]:
        """
        Uses LLM (via llm_interface) to validate if the question needs SQL,
        can be answered directly, or requires clarification. Uses instance history.
        """
        if not self.is_initialized:
            return {"action": "ERROR", "details": "Agent not initialized"}
        logging.info(f"Agent {self.agent_id} validating question: '{user_question}'")
        history_context = (
            self.conversation_history.get_formatted_history()
        )  # Use instance history

        analysis_context = "No detailed schema analysis available."
        if self.detailed_schema_analysis:
            analysis_context = (
                "\nDetailed Schema Analysis:\n"
                + json.dumps(self.detailed_schema_analysis, indent=2)
                + "\n"
            )

        return llm_interface.validate_question(
            question=user_question,
            schema=self.full_schema,
            analysis=analysis_context,
            history=history_context,
            llm=llm,
        )

    def _generate_sql_query(
        self, user_question: str, llm: llm_interface.LlmType
    ) -> Tuple[Optional[str], str]:
        """Generates SQL query using instance context, schema, history via llm_interface."""
        if not self.is_initialized:
            return None, "Agent not initialized"
        logging.info(
            f"Agent {self.agent_id} generating SQL for question: '{user_question}'"
        )
        if not self.query_engine:
            logging.error(f"Agent {self.agent_id}: Query engine not available.")
            return None, "Error: Query engine not initialized."

        try:
            # Use the shared query engine assigned to this instance
            retrieved_context = str(self.query_engine.query(user_question))
            logging.debug(
                f"Agent {self.agent_id}: Retrieved context for SQL generation: {retrieved_context}"
            )
        except Exception as e:
            logging.error(
                f"Agent {self.agent_id}: Error retrieving context from query engine: {e}",
                exc_info=True,
            )
            retrieved_context = "Context retrieval failed."

        history_context = (
            self.conversation_history.get_formatted_history()
        )  # Use instance history

        analysis_context = "No detailed schema analysis available."
        if self.detailed_schema_analysis:
            analysis_context = (
                "\nDetailed Schema Analysis:\n"
                + json.dumps(self.detailed_schema_analysis, indent=2)
                + "\n"
            )

        return llm_interface.generate_sql(
            question=user_question,
            schema=self.full_schema,
            analysis=analysis_context,
            history=history_context,
            context=retrieved_context,
            llm=llm,
        )

    def _execute_sql_query(self, sql_query: str) -> Union[List[Tuple], str]:
        """Executes the generated SQL query against the instance's DB cursor."""
        if not self.is_initialized:
            return "Agent not initialized"
        logging.info(f"Agent {self.agent_id} executing SQL: {sql_query}")
        if not self.cursor:
            logging.error(f"Agent {self.agent_id}: Database cursor is not initialized.")
            return "Error: Database connection is not available for this agent."
        try:
            self.cursor.execute(sql_query)
            results = self.cursor.fetchall()
            logging.info(
                f"Agent {self.agent_id}: Query executed successfully, {len(results)} rows returned."
            )
            return results
        except sqlite3.Error as e:
            logging.error(
                f"Agent {self.agent_id}: Error executing SQL query: {e}", exc_info=True
            )
            return f"Error executing query: {e}"
        except Exception as e:
            logging.error(
                f"Agent {self.agent_id}: An unexpected error occurred during query execution: {e}",
                exc_info=True,
            )
            return f"An unexpected error occurred: {e}"

    def _generate_analysis(
        self,
        user_question: str,
        sql_query: str,
        query_results: Union[List[Tuple], str],
        llm: llm_interface.LlmType,
    ) -> str:
        """Generates analysis using instance history and the instance's cursor."""
        if not self.is_initialized:
            return "Agent not initialized"
        logging.info(f"Agent {self.agent_id} generating analysis for query results.")
        history_context = (
            self.conversation_history.get_formatted_history()
        )  # Use instance history

        analysis_context = "No detailed schema analysis available."
        if self.detailed_schema_analysis:
            analysis_context = (
                "\nDetailed Schema Analysis:\n"
                + json.dumps(self.detailed_schema_analysis, indent=2)
                + "\n"
            )

        # Pass the instance's cursor
        return llm_interface.generate_analysis(
            question=user_question,
            sql=sql_query,
            query_results=query_results,
            schema=self.full_schema,
            analysis=analysis_context,
            history=history_context,
            llm=llm,
            cursor=self.cursor,  # Use the instance's cursor
        )

    def process_query(
        self, user_question: str, model_name: str
    ) -> Dict[str, Optional[str]]:
        """
        Processes the user's question for this specific agent instance.
        Uses the agent's conversation history and resources.
        """
        if not self.is_initialized or not self.query_engine or not self.conn:
            logging.error(
                f"Agent {self.agent_id} not initialized. Cannot process query."
            )
            return {
                "type": "error",
                "message": f"Agent {self.agent_id} is not ready.",
                "sql_query": None,
            }

        logging.info(
            f"Agent {self.agent_id} processing user query: '{user_question}' using model '{model_name}'"
        )

        # Get LLM instance (cached within this agent instance)
        try:
            llm = self._get_llm(model_name)  # Use the instance's LLM cache
            if not llm:
                raise ValueError(f"Could not initialize LLM for model: {model_name}")
        except Exception as e:
            logging.error(
                f"Agent {self.agent_id}: Failed to get LLM instance: {e}", exc_info=True
            )
            return {
                "type": "error",
                "message": f"Error initializing language model: {e}",
                "sql_query": None,
            }

        # --- Call agent methods (using instance history, schema, etc.) ---
        validation_result = self._validate_question(user_question, llm)
        action = validation_result["action"]
        details = validation_result["details"]

        response = {"type": action, "message": None, "sql_query": None}

        if action == "DIRECT_ANSWER":
            logging.info(f"Agent {self.agent_id}: Action=DIRECT_ANSWER")
            response["message"] = details
            self.conversation_history.add_interaction(  # Add to instance history
                question=user_question, response_type="direct_answer", analysis=details
            )

        elif action == "CLARIFICATION_NEEDED":
            logging.info(f"Agent {self.agent_id}: Action=CLARIFICATION_NEEDED")
            response["message"] = details
            self.conversation_history.add_interaction(  # Add to instance history
                question=user_question,
                response_type="clarification_needed",
                analysis=details,
            )

        elif action == "SQL_NEEDED":
            logging.info(f"Agent {self.agent_id}: Action=SQL_NEEDED")
            max_retries = 3  # Reduced retries slightly
            last_error_message = "An unknown error occurred during SQL processing."
            final_sql_query = None

            for attempt in range(max_retries):
                logging.info(
                    f"Agent {self.agent_id}: SQL Generation/Execution Attempt {attempt + 1}/{max_retries}"
                )

                sql_query, raw_llm_sql_response = self._generate_sql_query(
                    user_question, llm
                )
                final_sql_query = sql_query

                if not sql_query:
                    logging.error(
                        f"Agent {self.agent_id}: SQL generation failed on attempt {attempt + 1}."
                    )
                    last_error_message = (
                        raw_llm_sql_response
                        if raw_llm_sql_response
                        else "Error generating SQL query."
                    )
                    response["type"] = "error"
                    response["message"] = last_error_message
                    self.conversation_history.add_interaction(  # Add to instance history
                        question=user_question,
                        response_type="error",
                        analysis=last_error_message,
                    )
                    break  # Exit loop

                query_results = self._execute_sql_query(sql_query)

                if isinstance(query_results, str):  # Execution failed
                    error_message = query_results
                    logging.warning(
                        f"Agent {self.agent_id}: SQL execution failed on attempt {attempt + 1}: "
                        f"{error_message}."
                    )

                    # Generate analysis of the error for retry context
                    error_analysis_feedback = self._generate_analysis(
                        user_question, sql_query, error_message, llm
                    )
                    logging.info(
                        f"Agent {self.agent_id}: Analysis of execution error: {error_analysis_feedback}"
                    )

                    retry_context = (
                        f"The previous SQL query (`{sql_query}`) failed. "
                        f"Analysis: '{error_analysis_feedback}'. Generate a revised SQL query."
                    )
                    last_error_message = (
                        f"SQL Execution Error Analysis: {error_analysis_feedback}"
                    )

                    # Add error analysis context to this agent's history for the next attempt
                    self.conversation_history.add_interaction(
                        question=f"System: Analyzing execution error from attempt {attempt + 1}",
                        response_type="system_error_analysis_context",
                        sql_query=sql_query,
                        analysis=retry_context,
                    )

                    if attempt == max_retries - 1:
                        logging.error(
                            f"Agent {self.agent_id}: SQL query failed after {max_retries} attempts. "
                            f"Last error analysis: {error_analysis_feedback}"
                        )
                        final_error_message = (
                            f"Failed after {max_retries} attempts. "
                            f"Analysis: '{error_analysis_feedback}'. Last SQL: {sql_query}"
                        )
                        response["type"] = "error"
                        response["message"] = final_error_message
                        response["sql_query"] = sql_query
                        # Update instance history with final failure
                        self.conversation_history.add_interaction(
                            question=user_question,
                            response_type="error",
                            sql_query=sql_query,
                            analysis=response["message"],
                        )
                        break
                    else:
                        continue  # Go to the next attempt

                else:  # Execution successful
                    logging.info(
                        f"Agent {self.agent_id}: SQL execution successful on attempt {attempt + 1}."
                    )
                    # Optional: Add NULL check retry logic here if needed, similar to before.
                    # For simplicity, let's skip the automatic NULL check retry for now.

                    final_analysis = self._generate_analysis(
                        user_question, sql_query, query_results, llm
                    )
                    response["type"] = "sql_analysis"
                    response["message"] = final_analysis
                    response["sql_query"] = sql_query
                    self.conversation_history.add_interaction(  # Add to instance history
                        question=user_question,
                        response_type="sql_analysis",
                        sql_query=sql_query,
                        results=str(
                            query_results
                        ),  # Consider truncating results for history
                        analysis=final_analysis,
                    )
                    break  # Exit loop on success

            # Ensure final query is added if it exists and isn't already set
            if response.get("sql_query") is None and final_sql_query:
                response["sql_query"] = final_sql_query

        else:  # Should not happen
            logging.error(
                f"Agent {self.agent_id}: Invalid action '{action}' from validation."
            )
            response["type"] = "error"
            response["message"] = "Internal error during validation."
            self.conversation_history.add_interaction(  # Add to instance history
                question=user_question,
                response_type="error",
                analysis=response["message"],
            )

        logging.info(
            f"Agent {self.agent_id}: Finished processing query. Response type: {response['type']}"
        )
        return response

    def _get_llm(
        self, model_name: str
    ) -> Optional[Union[OpenAI, genai.GenerativeModel]]:
        """Gets an initialized LLM instance for THIS agent, caching it if necessary."""
        if model_name in self.initialized_llms:
            logging.debug(
                f"Agent {self.agent_id}: Using cached LLM instance for: {model_name}"
            )
            return self.initialized_llms[model_name]

        logging.info(f"Agent {self.agent_id}: Initializing LLM for: {model_name}")
        llm_instance: Optional[Union[OpenAI, genai.GenerativeModel]] = None
        try:
            # Reuse logic from llm_interface or centralize LLM creation if preferred
            if model_name.startswith("gpt-") or model_name.startswith("o3-"):
                if not os.environ.get("OPENAI_API_KEY"):
                    logging.error(
                        f"Agent {self.agent_id}: OpenAI API key missing for {model_name}."
                    )
                    return None
                llm_instance = OpenAI(
                    model=model_name,
                    temperature=0.01,
                    max_new_tokens=500,
                    request_timeout=60,
                )
            elif model_name.startswith("gemini-"):
                # Check if genai is configured (key/ADC) - should be done globally now
                # Rely on the genai library's internal configuration state
                try:
                    llm_instance = genai.GenerativeModel(model_name)
                except Exception as gemini_e:
                    logging.error(
                        f"Agent {self.agent_id}: Failed to initialize Gemini model {model_name}: "
                        f"{gemini_e}",
                        exc_info=True,
                    )
                    return None  # Return None if Gemini init fails

                # Check if API key / ADC seems available (optional check, library might handle)
                # if not os.getenv("GOOGLE_API_KEY") and not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
                #    logging.warning(
                #        f"Agent {self.agent_id}: Google API Key/Credentials may be missing "
                #        f"for {model_name}. Calls might fail."
                #    )
            else:
                logging.error(
                    f"Agent {self.agent_id}: Unsupported model name: {model_name}"
                )
                return None

            self.initialized_llms[model_name] = llm_instance
            logging.info(
                f"Agent {self.agent_id}: Successfully initialized LLM for: {model_name}"
            )
            return llm_instance

        except Exception as e:
            logging.error(
                f"Agent {self.agent_id}: Error initializing LLM {model_name}: {e}",
                exc_info=True,
            )
            return None

    def shutdown(self):
        """Closes DB connection for this agent instance."""
        logging.info(f"Shutting down agent instance {self.agent_id} resources...")
        if self.conn:
            try:
                self.conn.close()
                logging.info(f"Closed DB connection for agent {self.agent_id}.")
            except Exception as e:
                logging.error(
                    f"Error closing DB connection for agent {self.agent_id}: {e}"
                )
        self.conn = None
        self.cursor = None
        self.initialized_llms = {}  # Clear instance LLM cache
        # Don't nullify shared resources like query_engine, embed_model here
        self.is_initialized = False
        logging.info(f"Agent instance {self.agent_id} resources released.")


# Note: The main execution block (`if __name__ == "__main__":`)
# has been removed as this script will now be imported and used by the server.
# You would typically instantiate and use the TextToSqlAgent from your Flask app.
