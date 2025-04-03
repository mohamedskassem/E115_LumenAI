import os
import sqlite3
import json
import shutil # For deleting directories
import concurrent.futures
from typing import List, Tuple, Dict, Optional, Union
import logging
from llama_index.core import VectorStoreIndex, Document, Settings, PromptTemplate, StorageContext, load_index_from_storage
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.llms.gemini import Gemini

# ------------------------------
# Configure Logging
# ------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ------------------------------
# Load API Key from secrets
# ------------------------------
try:
    with open("secrets/openai_api_key.txt", "r") as key_file:
        os.environ["OPENAI_API_KEY"] = key_file.read().strip()
except FileNotFoundError:
    logging.error("API key file 'secrets/openai_api_key.txt' not found. Please create it.")
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
    
    def add_interaction(self, question: str, response_type: str, sql_query: Optional[str] = None, results: Optional[str] = None, analysis: Optional[str] = None):
        interaction = {
            "question": question,
            "response_type": response_type, # e.g., "sql_analysis", "direct_answer", "clarification_needed"
            "sql_query": sql_query,
            "results": results,
            "analysis": analysis
        }
        self.history.append(interaction)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        logging.debug(f"Added interaction to history. Current length: {len(self.history)}")
    
    def get_formatted_history(self) -> str:
        if not self.history:
            return ""
        
        formatted = "Previous conversation context:\n"
        for i, interaction in enumerate(self.history, 1):
            formatted += f"Interaction {i}:\n"
            formatted += f"  User Question: {interaction['question']}\n"
            if interaction['sql_query']:
                formatted += f"  SQL Generated: {interaction['sql_query']}\n"
            if interaction['results']:
                formatted += f"  Query Results: {interaction['results']}\n"
            if interaction['analysis']:
                formatted += f"  Response/Analysis: {interaction['analysis']}\n"
            formatted += "-" * 40 + "\n"
        return formatted

# ------------------------------
# TextToSqlAgent Class
# ------------------------------
class TextToSqlAgent:
    def __init__(self, db_path: str, cache_file: str = "schema_cache.json", default_model: str = "gpt-4-turbo"):
        logging.info("Initializing TextToSqlAgent object...")
        # Configuration details stored, but initialization deferred
        self.persist_dir = "vector_store_cache" # Directory to store the index
        self.db_path = db_path
        self.cache_file = cache_file
        self.default_model = default_model # Model for schema analysis and default queries
        self.default_llm = None # LLM instance used for schema loading
        self.initialized_llms: Dict[str, Union[OpenAI, Gemini]] = {} # Cache for dynamically loaded LLMs
        self.embed_model = None # Initialized in load_and_index
        self.index = None # Initialized in load_and_index
        self.query_engine = None # Initialized in load_and_index
        self.conn = None # Initialized in load_and_index
        self.cursor = None # Initialized in load_and_index
        self.full_schema = "" # Initialized in load_and_index
        self.conversation_history = ConversationHistory()
        self.is_initialized = False # Flag to track if indexing is complete
        # Defer actual initialization until load_and_index is called

    def load_and_index(self):
        """Loads data, initializes components, builds/loads index. Called after DB exists."""
        if self.is_initialized:
            logging.info("Agent already initialized.")
            return True
        
        if not os.path.exists(self.db_path):
            logging.error(f"Database path does not exist: {self.db_path}. Cannot initialize.")
            return False
            
        logging.info(f"Starting initialization and indexing for DB: {self.db_path}")
        logging.info("Initializing LLM and Embeddings...")
        try:
            # Initialize the DEFAULT LLM (used for schema analysis)
            # Note: We don't set the global Settings.llm anymore
            self.default_llm = self._get_llm(self.default_model) # Initialize default model immediately
            if not self.default_llm:
                raise ValueError(f"Failed to initialize default LLM: {self.default_model}")
            logging.info(f"Default LLM ({self.default_model}) initialized for agent startup tasks.")

            # Initialize Embeddings
            logging.info("Initializing OpenAI Embeddings (text-embedding-3-small)...")
            self.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        except Exception as e:
            logging.error(f"Error initializing LLM or Embeddings: {e}", exc_info=True)
            raise  # Re-raise after logging

        logging.info("Loading DB schema and column summaries...")
        try:
            # Load database and generate schema documents
            self.full_schema, schema_docs, self.conn = self._load_db_schema_and_column_summaries()
            self.cursor = self.conn.cursor()
            logging.info("Database schema loaded and connection established.")
            logging.debug(f"Full Schema:\n{self.full_schema}")
        except Exception as e:
            logging.error(f"Error loading database schema: {e}", exc_info=True)
            if self.conn:
                self.conn.close()
            raise

        logging.info("Initializing Vector Store Index...")
        # --- Try Loading Index from Cache --- 
        if os.path.exists(self.persist_dir):
            if not self.embed_model: # Ensure embed model is initialized first
                logging.error("Embed model not initialized before loading index.")
                return False
            try:
                logging.info(f"Loading existing vector store index from: {self.persist_dir}")
                storage_context = StorageContext.from_defaults(persist_dir=self.persist_dir)
                self.index = load_index_from_storage(storage_context, embed_model=self.embed_model)
                self.query_engine = self.index.as_query_engine(response_mode="no_text")
                logging.info("Vector Store Index loaded successfully from cache.")
            except Exception as e:
                logging.warning(f"Failed to load index from cache ({self.persist_dir}): {e}. Rebuilding...", exc_info=True)
                self._build_and_persist_index(schema_docs) # Fallback to building
        else:
             logging.info(f"No existing vector store cache found at {self.persist_dir}. Building new index...")
             self._build_and_persist_index(schema_docs)

        self.is_initialized = True
        logging.info("Agent initialization and indexing complete.")
        return True

    def _build_and_persist_index(self, schema_docs: List[Document]):
        """Builds the vector index from documents and persists it."""
        try:
            # Temporarily set global LLM for index/engine setup, if needed by components
            # to prevent falling back to implicit OpenAI defaults.
            original_llm = Settings.llm # Store original global (if any)
            Settings.llm = self.default_llm # Use the agent's default LLM

            # Initialize LlamaIndex with documents
            self.index = VectorStoreIndex.from_documents(schema_docs, embed_model=self.embed_model)
            # Configure query engine to only retrieve context, not synthesize text
            self.query_engine = self.index.as_query_engine(response_mode="no_text")
            
            logging.info(f"Persisting index to: {self.persist_dir}")
            self.index.storage_context.persist(persist_dir=self.persist_dir)

            # Restore original global LLM setting after setup
            Settings.llm = original_llm

            # self._print_vector_store_samples(num_samples=1) # Optional: for debugging
            logging.info("Vector Store Index built and persisted successfully.")
        except Exception as e:
            logging.error(f"Error initializing Vector Store Index: {e}", exc_info=True)
            if self.conn:
                self.conn.close()
            raise

    def _analyze_column(self, col_name, table_name, sample_data, llm):
        """Analyzes a single column using the provided LLM instance."""
        combined_prompt = f"""
        Briefly describe column '{col_name}' in table '{table_name}':
        - Purpose
        - Data type
        - Sample values: {sample_data[:3]}

        Keep the response under 2 sentences.
        """
        try:
            response = llm.predict(PromptTemplate(template=combined_prompt))
            doc_text = (
                f"Table: {table_name}\n"
                f"Column: {col_name}\n"
                f"Summary: {response.strip()}"
            )
            return doc_text
        except Exception as e:
            logging.error(f"Error analyzing column {table_name}.{col_name}: {e}", exc_info=True)
            # Return a basic doc text even if LLM fails
            return f"Table: {table_name}\nColumn: {col_name}\nSummary: Analysis failed."

    def _process_table(self, table_info: Tuple[str, List[Tuple], List[Tuple], dict], llm) -> Tuple[str, List[Document]]:
        """Processes a single table and its columns."""
        table_name, columns, sample_data, column_indices = table_info
        table_schema = f"Table: {table_name}\n"
        schema_documents = []
        logging.debug(f"Processing table: {table_name}")

        for col in columns:
            col_name = col[1]
            col_type = col[2]
            table_schema += f"  - {col_name} ({col_type})\n"
            
            col_samples = [row[column_indices[col_name]] for row in sample_data if len(row) > column_indices[col_name]]
            
            doc_text = self._analyze_column(col_name, table_name, col_samples, llm)
            schema_documents.append(Document(text=doc_text))
        
        table_schema += "\n"
        return table_schema, schema_documents

    def _load_db_schema_and_column_summaries(self) -> Tuple[str, List[Document], sqlite3.Connection]:
        """Loads database schema and generates summaries with parallel processing and caching."""
        if os.path.exists(self.cache_file):
            logging.info(f"Loading cached schema analysis from {self.cache_file}...")
            try:
                with open(self.cache_file, 'r') as f:
                    cache = json.load(f)
                    conn = sqlite3.connect(self.db_path, check_same_thread=False) # Ensure thread safety for Flask
                    return cache['schema'], [Document(text=doc) for doc in cache['documents']], conn
            except (json.JSONDecodeError, KeyError, sqlite3.Error) as e:
                 logging.warning(f"Failed to load or parse cache file {self.cache_file}: {e}. Re-generating schema.", exc_info=True)
                 # If cache is invalid, proceed to generate fresh schema

        conn = sqlite3.connect(self.db_path, check_same_thread=False) # Ensure thread safety for Flask
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        logging.info(f"Found {len(tables)} tables. Analyzing schema...")

        table_data = []
        for table in tables:
            table_name = table[0]
            try:
                cursor.execute(f"PRAGMA table_info(\"{table_name}\");") # Use quotes for safety
                columns = cursor.fetchall()
                cursor.execute(f"SELECT * FROM \"{table_name}\" LIMIT 50;")
                sample_data = cursor.fetchall()
                column_indices = {col[1]: idx for idx, col in enumerate(columns)}
                table_data.append((table_name, columns, sample_data, column_indices))
            except sqlite3.Error as e:
                logging.error(f"Error fetching schema or data for table {table_name}: {e}", exc_info=True)
                # Continue with other tables if one fails

        full_schema = ""
        schema_documents = []
        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(5, len(table_data) if table_data else 1)) as executor:
            future_to_table = {
                # Use the default LLM for schema analysis during startup
                executor.submit(self._process_table, table_info, self.default_llm): table_info[0]
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

        # Save to cache
        try:
            cache = {
                'schema': full_schema,
                'documents': [doc.text for doc in schema_documents]
            }
            with open(self.cache_file, 'w') as f:
                json.dump(cache, f, indent=4)
            logging.info(f"Schema analysis complete and saved to {self.cache_file}.")
        except IOError as e:
            logging.error(f"Error saving cache file {self.cache_file}: {e}", exc_info=True)

        return full_schema, schema_documents, conn

    def _validate_question(self, user_question: str, llm) -> Dict:
        """
        Uses LLM to validate if the question needs SQL, can be answered directly,
        or requires clarification.
        """
        logging.info(f"Validating question: '{user_question}'")
        history_context = self.conversation_history.get_formatted_history()

        validation_prompt_str = f"""
You are an AI assistant helping determine how to answer a user's question based on available database schema and conversation history.

Database Schema Summary:
{self.full_schema}

Conversation History:
{history_context}

User Question: "{user_question}"

Analyze the user question in the context of the schema and history. Determine ONE of the following actions:

1.  **SQL_NEEDED**: The question requires querying the database.
2.  **DIRECT_ANSWER**: The question can be answered directly using the provided schema summary, conversation history, or general knowledge (e.g., it's a greeting or a question about the AI itself). If choosing this, provide the direct answer.
3.  **CLARIFICATION_NEEDED**: The question is ambiguous, lacks specifics needed for a query (e.g., needs date ranges, specific IDs), or refers to information clearly not in the schema or history. If choosing this, suggest what clarification is needed.

Respond ONLY with the chosen action label (SQL_NEEDED, DIRECT_ANSWER, or CLARIFICATION_NEEDED) followed by a colon and the answer/clarification if applicable.

Examples:
User Question: "What are the total sales for product ID 5?"
OUTPUT: SQL_NEEDED:

User Question: "Hello there!"
OUTPUT: DIRECT_ANSWER: Hello! How can I help you with the Adventure Works data today?

User Question: "Show me the recent orders."
OUTPUT: CLARIFICATION_NEEDED: Could you please specify what you mean by 'recent'? For example, provide a date range (like 'last month' or 'since January 1st, 2024').

User Question: "Can you tell me about the company CEO?"
OUTPUT: DIRECT_ANSWER: I can only answer questions about the data in the Adventure Works database schema provided. I don't have information about the company's personnel like the CEO.

User Question: "What tables do we have?"
OUTPUT: DIRECT_ANSWER: The database contains tables like: [List a few table names from the schema].

Now, analyze the current user question.

OUTPUT:""" # Ensure the LLM starts its response right after this.

        try:
            prompt = PromptTemplate(template=validation_prompt_str)
            response = llm.predict(prompt).strip()
            logging.debug(f"Validation LLM response: {response}")

            parts = response.split(":", 1)
            action = parts[0].strip()
            details = parts[1].strip() if len(parts) > 1 else ""

            if action in ["SQL_NEEDED", "DIRECT_ANSWER", "CLARIFICATION_NEEDED"]:
                return {"action": action, "details": details}
            else:
                # Handle unexpected LLM response format
                logging.warning(f"Unexpected validation response format: {response}. Defaulting to SQL_NEEDED.")
                # Fallback: Assume SQL is needed if unsure, or maybe ask for clarification.
                # Let's default to SQL_NEEDED as it was the original behavior.
                return {"action": "SQL_NEEDED", "details": ""}
        except Exception as e:
            logging.error(f"Error during question validation LLM call: {e}", exc_info=True)
            # Fallback in case of API error
            return {"action": "SQL_NEEDED", "details": ""} # Default to SQL needed on error

    def _generate_sql_query(self, user_question: str, llm) -> Tuple[Optional[str], str]:
        """Generates SQL query using context from LlamaIndex and schema information."""
        logging.info(f"Generating SQL for question: '{user_question}'")
        try:
            retrieved_context = str(self.query_engine.query(user_question))
            logging.debug(f"Retrieved context for SQL generation: {retrieved_context}")
        except Exception as e:
             logging.error(f"Error retrieving context from query engine: {e}", exc_info=True)
             retrieved_context = "Context retrieval failed." # Provide fallback context

        history_context = self.conversation_history.get_formatted_history()

        developer_msg = f"""You are an expert SQL generator. You will be provided with a user query, database schema, conversation history, and retrieved context.
Your goal is to generate a single, valid SQL query (compatible with SQLite) to provide the best answer to the user's most recent question.

Database Schema:
{self.full_schema}

Conversation History:
{history_context}

Retrieved Context (Information potentially relevant to the user query):
{retrieved_context}

Based on all the above, generate ONLY the SQL query for the following user input.
- Do not include explanations or 'OUTPUT:'. Just the SQL.
- For questions asking for the "top N", "highest", "lowest", "most", or "least", use ORDER BY with DESC or ASC and LIMIT N.

Examples:

-- Simple Query --
USER INPUT: Show me the total revenue for each region for sales made in 2024.
GENERATED SQL: SELECT Region, SUM(Quantity * Price) AS TotalRevenue FROM Sales WHERE Date BETWEEN '2024-01-01' AND '2024-12-31' GROUP BY Region;

-- Complex Query (Top N) --
USER INPUT: What are the top 3 products with the most sales quantity in 2023?
GENERATED SQL: SELECT p.ProductName, SUM(s.OrderQuantity) AS TotalQuantity
FROM AdventureWorks_Sales_2023 s
JOIN AdventureWorks_Products p ON s.ProductKey = p.ProductKey
GROUP BY p.ProductName
ORDER BY TotalQuantity DESC
LIMIT 3;
"""

        user_msg = f"USER INPUT: {user_question}"
        prompt_str = developer_msg + "\n" + user_msg + "\nGENERATED SQL:" # Guide the LLM output

        try:
            prompt = PromptTemplate(template=prompt_str)
            logging.info(f"Attempting SQL generation using LLM type: {type(llm)}") # DEBUG LOG
            response = llm.predict(prompt).strip()
            logging.info(f"Completed SQL generation using LLM type: {type(llm)}") # DEBUG LOG
            logging.debug(f"SQL Generation LLM response: {response}")

            # Basic validation/cleanup of the generated SQL
            final_sql = response
            # Remove potential markdown backticks
            if final_sql.startswith("```sql"):
                final_sql = final_sql[len("```sql"):].strip()
            if final_sql.endswith("```"):
                final_sql = final_sql[:-len("```")].strip()
            # Remove potential leading/trailing semicolons if they cause issues, though usually fine for execute()
            final_sql = final_sql.strip().rstrip(';')

            logging.info(f"Generated SQL: {final_sql}")
            return final_sql, response # Return both raw and cleaned
        except Exception as e:
            logging.error(f"Error during SQL generation LLM call with {type(llm)}: {e}", exc_info=True) # DEBUG LOG
            return None, f"Error generating SQL: {e}"

    def _execute_sql_query(self, sql_query: str) -> Union[List[Tuple], str]:
        """Executes the generated SQL query against the database."""
        logging.info(f"Executing SQL: {sql_query}")
        if not self.cursor:
             logging.error("Database cursor is not initialized.")
             return "Error: Database connection is not available."
        try:
            self.cursor.execute(sql_query)
            results = self.cursor.fetchall()
            # Limit results for display/analysis if necessary
            # if len(results) > 100:
            #     logging.warning(f"Query returned {len(results)} rows. Truncating for analysis.")
            #     # return results[:100] # Optionally truncate here or in analysis prompt
            logging.info(f"Query executed successfully, {len(results)} rows returned.")
            return results
        except sqlite3.Error as e:
            logging.error(f"Error executing SQL query: {e}", exc_info=True)
            return f"Error executing query: {e}"
        except Exception as e: # Catch other potential errors
            logging.error(f"An unexpected error occurred during query execution: {e}", exc_info=True)
            return f"An unexpected error occurred: {e}"

    def _generate_analysis(self, user_question: str, sql_query: str, query_results: Union[List[Tuple], str], llm) -> str:
        """Generates a natural language analysis of the query results."""
        logging.info("Generating analysis for query results.")
        history_context = self.conversation_history.get_formatted_history()

        # Format results nicely for the prompt
        if isinstance(query_results, str): # Error message
            formatted_results = query_results
        elif not query_results:
             formatted_results = "The query returned no results."
        else:
            # Convert results to a more readable string format, maybe limit rows
            headers = [desc[0] for desc in self.cursor.description] if self.cursor.description else []
            results_str = "Headers: " + ", ".join(headers) + "\n"
            results_str += "\n".join([str(row) for row in query_results[:20]]) # Limit rows in prompt
            if len(query_results) > 20:
                results_str += f"\n... (truncated, {len(query_results)} total rows)"
            formatted_results = results_str

        analysis_prompt_str = f"""
You are LumenAI, a helpful data analyst AI. Your goal is to provide a clear, concise, and natural language response to the user's question based on the executed SQL query and its results. Incorporate context from the conversation history if relevant.

Conversation History:
{history_context}

Most Recent Interaction:
- User Question: {user_question}
- SQL Query Executed: {sql_query}
- Query Results:
{formatted_results}

Guidelines for your response:
1.  **Directly Address the Question**: Start by answering the user's original question based on the results.
2.  **Summarize Key Findings**: Briefly explain what the data shows.
3.  **Format Clearly**: Use formatting (like lists or bolding) if it improves readability. Format numbers understandably (e.g., use commas).
4.  **Contextualize (If Applicable)**: If the conversation history provides relevant context (e.g., comparing to a previous query), mention it briefly (e.g., "Compared to last month...", "This is an increase from...").
5.  **Handle Errors/No Results Gracefully**: If the results indicate an error or are empty, state that clearly and perhaps suggest alternatives or checking the query.
6.  **Be Concise**: Keep the analysis focused and avoid unnecessary jargon.
7.  **Conversational Tone**: Maintain a helpful and professional yet conversational style.

Example (Good Analysis):
"Based on the data, the total sales for the 'Bikes' category in 2023 amounted to $1,234,567. This represents a 15% increase compared to the $1,073,536 in sales for 2022."

Example (Handling No Results):
"The query didn't find any sales records for Product ID 999 in the specified date range."

Example (Handling Error):
"There was an error when trying to run the query: [Error message]. This might be due to an issue with the generated SQL. Would you like me to try rephrasing the query?"

Now, generate the analysis for the user:
"""

        try:
            analysis_prompt = PromptTemplate(template=analysis_prompt_str)
            analysis = llm.predict(analysis_prompt).strip()
            logging.info("Analysis generated successfully.")
            logging.debug(f"Generated Analysis: {analysis}")
            return analysis
        except Exception as e:
            logging.error(f"Error during analysis generation LLM call: {e}", exc_info=True)
            return f"Error generating analysis: {e}"

    def process_query(self, user_question: str, model_name: str) -> Dict[str, Optional[str]]:
        """
        Processes the user's question: validates, potentially clarifies,
        generates/executes SQL, and returns analysis or response. Requires agent to be initialized.
        """
        if not self.is_initialized or not self.query_engine or not self.conn:
            logging.error("Agent not initialized. Cannot process query.")
            return {"type": "error", "message": "Agent is not ready. Please ensure data is loaded and indexed.", "sql_query": None}

        logging.info(f"Processing user query: '{user_question}'")

        # Get the appropriate LLM instance for this request
        try:
            llm = self._get_llm(model_name)
            if not llm:
                 raise ValueError(f"Could not initialize LLM for model: {model_name}")
        except Exception as e:
             logging.error(f"Failed to get LLM instance: {e}", exc_info=True)
             return {"type": "error", "message": f"Error initializing language model: {e}", "sql_query": None}

        validation_result = self._validate_question(user_question, llm)
        action = validation_result["action"]
        details = validation_result["details"]

        response = {"type": action, "message": None, "sql_query": None}

        if action == "DIRECT_ANSWER":
            logging.info("Action: DIRECT_ANSWER")
            response["message"] = details
            self.conversation_history.add_interaction(
                question=user_question,
                response_type="direct_answer",
                analysis=details
            )

        elif action == "CLARIFICATION_NEEDED":
            logging.info("Action: CLARIFICATION_NEEDED")
            response["message"] = details
            # Don't add to history until clarification is resolved, or add with a specific type
            self.conversation_history.add_interaction(
                 question=user_question,
                 response_type="clarification_needed",
                 analysis=details # Store the clarification question asked
            )

        elif action == "SQL_NEEDED":
            logging.info("Action: SQL_NEEDED")
            sql_query, raw_llm_sql_response = self._generate_sql_query(user_question, llm)
            response["sql_query"] = sql_query # Include the generated SQL in the response

            if sql_query:
                query_results = self._execute_sql_query(sql_query)
                analysis = self._generate_analysis(user_question, sql_query, query_results, llm)
                response["message"] = analysis
                self.conversation_history.add_interaction(
                    question=user_question,
                    response_type="sql_analysis",
                    sql_query=sql_query,
                    results=str(query_results) if not isinstance(query_results, str) else query_results, # Store results as string
                    analysis=analysis
                )
            else:
                logging.error("SQL generation failed.")
                response["message"] = "I encountered an error trying to generate the SQL query needed to answer your question."
                # Add error interaction to history
                self.conversation_history.add_interaction(
                    question=user_question,
                    response_type="error",
                    analysis=response["message"]
                 )

        else: # Should not happen based on _validate_question logic
            logging.error(f"Invalid action '{action}' received from validation.")
            response["type"] = "error"
            response["message"] = "An unexpected internal error occurred during question validation."
            self.conversation_history.add_interaction(
                question=user_question,
                response_type="error",
                analysis=response["message"]
            )

        logging.info(f"Finished processing query. Response type: {response['type']}")
        return response

    def _get_llm(self, model_name: str) -> Optional[Union[OpenAI, Gemini]]:
        """Gets an initialized LLM instance, caching it if necessary."""
        if model_name in self.initialized_llms:
            logging.debug(f"Using cached LLM instance for: {model_name}")
            return self.initialized_llms[model_name]

        logging.info(f"Initializing LLM for: {model_name}")
        llm_instance = None
        try:
            if model_name.startswith("gpt-"):
                llm_instance = OpenAI(
                    model=model_name,
                    temperature=0.01,
                    max_new_tokens=500, # Consider adjusting based on model
                    request_timeout=60
                )
            # Handle both standard (models/gemini-...) and experimental Gemini names
            elif model_name.startswith("models/gemini-") or model_name == "gemini-2.5-pro-exp-03-25":
                # When using Service Account authentication (ADC), the library primarily uses GOOGLE_APPLICATION_CREDENTIALS.
                # We just need to ensure that variable is likely set (it's set in dockershell.sh).
                if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
                    logging.error("GOOGLE_APPLICATION_CREDENTIALS environment variable not set. Cannot initialize Gemini via ADC.")
                    return None

                # Construct the correct model name if it doesn't start with models/
                # The library requires the 'models/' prefix even for experimental names
                # For experimental ones, it might accept the short name directly.
                # Let's pass the provided name directly for now.
                # If this causes issues, we might need to prepend "models/"
                # conditionally based on the specific experimental model string.
                gemini_model_id = model_name
                # The library requires the 'models/' prefix even for experimental names
                if not gemini_model_id.startswith("models/") and not gemini_model_id.startswith("tunedModels/"):
                     gemini_model_id = f"models/{gemini_model_id}"
                     logging.info(f"Prepended 'models/' prefix. Using: {gemini_model_id}")

                llm_instance = Gemini(
                    model_name=gemini_model_id,
                    temperature=0.01, # Note: Gemini might use different temp scale/defaults
                    # Add other relevant Gemini parameters if needed
                )
            else:
                logging.error(f"Unsupported model name provided: {model_name}")
                return None

            self.initialized_llms[model_name] = llm_instance
            logging.info(f"Successfully initialized LLM for: {model_name}")
            return llm_instance

        except Exception as e:
            logging.error(f"Error initializing LLM {model_name}: {e}", exc_info=True)
            return None # Failed to initialize

    def _print_vector_store_samples(self, num_samples=3):
        """Prints sample documents from the vector store."""
        if not self.index or not self.index.docstore:
            logging.warning("Vector store not initialized or empty. Cannot print samples.")
            return

        logging.debug("\n" + "="*80)
        logging.debug("VECTOR STORE SAMPLES:")
        logging.debug("="*80)
        try:
            documents = list(self.index.docstore.docs.values())
            if not documents:
                logging.debug("No documents found in the vector store.")
                return

            for i, doc in enumerate(documents[:num_samples]):
                logging.debug(f"\nSample Document {i+1}:")
                logging.debug("-"*40)
                logging.debug(doc.text)
                logging.debug("-"*40)

            logging.debug("\n" + "="*80 + "\n")
        except Exception as e:
            logging.error(f"Error accessing vector store documents: {e}", exc_info=True)

    def shutdown(self):
        """Closes DB connection and resets state."""
        logging.info("Shutting down agent resources...")
        if self.conn:
            self.conn.close()
            self.conn = None
        self.cursor = None
        self.index = None
        self.query_engine = None
        self.default_llm = None
        self.initialized_llms = {}
        self.embed_model = None
        self.full_schema = ""
        self.is_initialized = False
        # Keep conversation history? Optional. Let's keep it for now.
        logging.info("Agent resources released.")

# Note: The main execution block (`if __name__ == "__main__":`)
# has been removed as this script will now be imported and used by the server.
# You would typically instantiate and use the TextToSqlAgent from your Flask app.