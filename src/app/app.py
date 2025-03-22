import os
import sqlite3
import json
import concurrent.futures
from typing import List, Tuple, Dict
from llama_index.core import VectorStoreIndex, Document, Settings, PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI

# ------------------------------
# Load API Key from secrets
# ------------------------------
with open("secrets/openai_api_key.txt", "r") as key_file:
    os.environ["OPENAI_API_KEY"] = key_file.read().strip()

# Disable tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ------------------------------
# Conversation History Class
# ------------------------------
class ConversationHistory:
    def __init__(self, max_history: int = 5):
        self.history: List[Dict] = []
        self.max_history = max_history
    
    def add_interaction(self, question: str, sql_query: str, results: str, analysis: str):
        interaction = {
            "question": question,
            "sql_query": sql_query,
            "results": results,
            "analysis": analysis
        }
        self.history.append(interaction)
        if len(self.history) > self.max_history:
            self.history.pop(0)
    
    def get_formatted_history(self) -> str:
        if not self.history:
            return ""
        
        formatted = "\nPrevious conversation context:\n"
        for i, interaction in enumerate(self.history, 1):
            formatted += f"\nQ{i}: {interaction['question']}\n"
            formatted += f"SQL: {interaction['sql_query']}\n"
            formatted += f"Results: {interaction['results']}\n"
            formatted += f"Analysis: {interaction['analysis']}\n"
            formatted += "-" * 40 + "\n"
        return formatted

# ------------------------------
# Function: Analyze Column
# ------------------------------
def analyze_column(col_name, table_name, sample_data, llm):
    """
    Optimized column analysis with a more concise prompt
    """
    combined_prompt = f"""
    Briefly describe column '{col_name}' in table '{table_name}':
    - Purpose
    - Data type
    - Sample values: {sample_data[:3]}  # Reduced from 5 to 3 samples
    
    Keep the response under 2 sentences.
    """
    
    response = llm.predict(PromptTemplate(template=combined_prompt))
    
    # Create a minimal document structure
    doc_text = (
        f"Table: {table_name}\n"
        f"Column: {col_name}\n"
        f"Summary: {response.strip()}"
    )
    
    return doc_text

# ------------------------------
# Function: Process Table
# ------------------------------
def process_table(table_info: Tuple[str, List[Tuple], List[Tuple], dict], llm) -> Tuple[str, List[Document]]:
    """
    Process a single table and its columns
    """
    table_name, columns, sample_data, column_indices = table_info
    table_schema = f"Table: {table_name}\n"
    schema_documents = []
    
    for col in columns:
        col_name = col[1]
        col_type = col[2]
        table_schema += f"  - {col_name} ({col_type})\n"
        
        # Get sample values for this column
        col_samples = [row[column_indices[col_name]] for row in sample_data]
        
        # Perform column analysis
        doc_text = analyze_column(col_name, table_name, col_samples, llm)
        schema_documents.append(Document(text=doc_text))
    
    table_schema += "\n"
    return table_schema, schema_documents

# ------------------------------
# Function: Load DB Schema and Column Summaries
# ------------------------------
def load_db_schema_and_column_summaries(db_path: str, llm, cache_file: str = "schema_cache.json") -> Tuple[str, List[Document], sqlite3.Connection]:
    """
    Load database schema and generate summaries with parallel processing
    """
    # Try to load from cache first
    if os.path.exists(cache_file):
        print("Loading cached schema analysis...")
        with open(cache_file, 'r') as f:
            cache = json.load(f)
            return cache['schema'], [Document(text=doc) for doc in cache['documents']], sqlite3.connect(db_path)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    print("\nLoading database schema and analyzing columns...")
    
    # Prepare table data for parallel processing
    table_data = []
    for table in tables:
        table_name = table[0]
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 50")  # Reduced from 100 to 50
        sample_data = cursor.fetchall()
        
        column_indices = {col[1]: idx for idx, col in enumerate(columns)}
        table_data.append((table_name, columns, sample_data, column_indices))
    
    # Process tables in parallel
    full_schema = ""
    schema_documents = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_to_table = {
            executor.submit(process_table, table_info, llm): table_info[0]
            for table_info in table_data
        }
        
        for future in concurrent.futures.as_completed(future_to_table):
            table_name = future_to_table[future]
            try:
                table_schema, table_docs = future.result()
                full_schema += table_schema
                schema_documents.extend(table_docs)
                print(f"Completed processing table: {table_name}")
            except Exception as e:
                print(f"Error processing table {table_name}: {e}")
    
    # Save to cache
    cache = {
        'schema': full_schema,
        'documents': [doc.text for doc in schema_documents]
    }
    with open(cache_file, 'w') as f:
        json.dump(cache, f)
    
    print("\nSchema analysis complete!")
    return full_schema, schema_documents, conn

# ------------------------------
# Function: Generate SQL Query
# ------------------------------
def generate_sql_query(user_question, full_schema, query_engine, llm, conversation_history=None):
    """
    Generate SQL query using context from LlamaIndex and schema information.
    Now includes conversation history for context.
    """
    # Retrieve relevant context from LlamaIndex
    retrieved_context = str(query_engine.query(user_question))

    # Get conversation history if available
    history_context = ""
    if conversation_history:
        history_context = conversation_history.get_formatted_history()

    # Construct the prompt with developer instructions and example
    developer_msg = (
        "You will be provided with a user query.\n"
        "Your goal is to generate a valid SQL query to provide the best answer to the user.\n\n"
        "This is the table schema:\n"
        f"{full_schema}\n\n"
        f"{history_context}\n"  # Add conversation history
        "Use this schema to generate as an output the SQL query.\n\n"
        "For example:\n\n"
        "USER INPUT: Show me the total revenue for each region for sales made in 2024.\n"
        "OUTPUT: SELECT Region, SUM(Quantity * Price) AS TotalRevenue\n"
        "FROM Sales\n"
        "WHERE Date BETWEEN '2024-01-01' AND '2024-12-31'\n"
        "GROUP BY Region;\n\n"
    )
    
    # Combine all context and user question
    user_msg = f"USER INPUT: {user_question}"
    prompt_str = developer_msg + "Retrieved Context:\n" + retrieved_context + "\n\n" + user_msg

    # Generate SQL query using LLM
    prompt = PromptTemplate(template=prompt_str)
    response = llm.predict(prompt)
    
    # Extract SQL query from response
    final_sql = None
    for line in response.splitlines():
        if line.strip().startswith("OUTPUT:"):
            final_sql = line.strip()[len("OUTPUT:"):].strip()
            break
    if final_sql is None:
        final_sql = response.strip()
        
    return final_sql, response

# ------------------------------
# Function: Generate Analysis
# ------------------------------
def generate_analysis(sql_query, query_results, full_schema, llm, user_question, conversation_history=None):
    """
    Generate analysis of query results in the context of the original question.
    Now includes comparative analysis with previous interactions.
    """
    # Get relevant previous interaction for comparison
    previous_context = ""
    if conversation_history and conversation_history.history:
        previous_interaction = conversation_history.history[-1]
        previous_context = f"""
Previous interaction:
Question: {previous_interaction['question']}
Results: {previous_interaction['results']}
"""

    analysis_prompt_str = f"""
You are a helpful data analyst having a conversation with a user. Analyze the query results and provide a natural, conversational response.

Context:
- User Question: {user_question}
- Query Results: {query_results}
{previous_context}

Guidelines for your response:
1. Start with a direct answer to the question
2. Format numbers in a readable way (e.g., 1,234,567 instead of 1234567)
3. If there's previous context:
   - Compare with previous results
   - Highlight any interesting changes or trends
   - Use phrases like "compared to", "increased/decreased by", "higher/lower than"
4. Add relevant insights or observations
5. Keep the tone conversational but professional
6. If the results show a significant change, suggest possible factors

Example style:
"The sales for 2017 were 47,000 units, which is 2,000 units higher than 2016's 45,000 units. This 4.4% increase might be attributed to the new product launches in Q2 2017."

Please provide your analysis:
"""
    
    analysis_prompt = PromptTemplate(template=analysis_prompt_str)
    return llm.predict(analysis_prompt)

# ------------------------------
# Function: Print Vector Store Samples
# ------------------------------
def print_vector_store_samples(index, num_samples=3):
    """
    Print sample documents from the vector store to see what information is stored.
    """
    print("\n" + "="*80)
    print("VECTOR STORE SAMPLES:")
    print("="*80)
    
    # Get all documents from the index
    documents = index.docstore.docs.values()
    
    # Print first num_samples documents
    for i, doc in enumerate(list(documents)[:num_samples]):
        print(f"\nSample Document {i+1}:")
        print("-"*40)
        print(doc.text)
        print("-"*40)
    
    print("\n" + "="*80 + "\n")

# ------------------------------
# Main Application
# ------------------------------
def main():
    # Initialize LLM with optimized settings
    Settings.llm = OpenAI(
        temperature=0.01,
        max_new_tokens=500,
        model="o3-mini",
        request_timeout=30
    )
    llm = Settings.llm

    # Initialize conversation history
    conversation_history = ConversationHistory()

    # Load database and generate schema documents
    db_path = "./output/adventure_works.db"
    full_schema, schema_docs, conn = load_db_schema_and_column_summaries(db_path, llm)

    # Initialize LlamaIndex with documents
    embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
    index = VectorStoreIndex.from_documents(schema_docs, embed_model=embed_model)
    query_engine = index.as_query_engine()

    # Print samples from vector store
    print_vector_store_samples(index)

    # Initialize database cursor
    cursor = conn.cursor()
    print("SQLite database loaded successfully.\n")
    print("Database Schema:")
    print(full_schema)

    # Interactive query loop
    while True:
        user_question = input("Ask any question about the data (or type 'exit' to quit): ")
        if user_question.lower() in ["exit", "quit"]:
            break

        # Generate and execute query with conversation history
        final_sql_query, full_response = generate_sql_query(
            user_question, 
            full_schema, 
            query_engine, 
            llm,
            conversation_history
        )
        print("\nGenerated SQL Query:")
        print(final_sql_query)

        # Execute query and handle results
        try:
            cursor.execute(final_sql_query)
            query_results = cursor.fetchall()
        except Exception as e:
            query_results = f"Error executing query: {e}"
        print("\nQuery Results:")
        print(query_results)

        # Generate analysis of results with conversation history
        analysis = generate_analysis(
            final_sql_query, 
            query_results, 
            full_schema, 
            llm, 
            user_question,
            conversation_history
        )
        print("\nAnalysis:")
        print(analysis)
        print("\n" + "-" * 60 + "\n")

        # Add the interaction to conversation history
        conversation_history.add_interaction(
            user_question,
            final_sql_query,
            str(query_results),
            analysis
        )

    # Cleanup
    conn.close()
    print("Application exited.")

if __name__ == "__main__":
    main()