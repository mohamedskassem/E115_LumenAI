import os
import sqlite3
from llama_index.core import VectorStoreIndex, Document, Settings, PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI

# ------------------------------
# Load API Key from secrets
# ------------------------------
with open("secrets/openai_api_key.txt", "r") as key_file:
    os.environ["OPENAI_API_KEY"] = key_file.read().strip()

# Disable tokenizer parallelism warnings.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ------------------------------
# Function: Load DB Schema
# ------------------------------
def load_db_schema(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    full_schema = ""
    schema_documents = []
    for table in tables:
        table_name = table[0]
        table_schema = f"Table: {table_name}\n"
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        for col in columns:
            table_schema += f"  - {col[1]} ({col[2]})\n"
        table_schema += "\n"
        full_schema += table_schema
        schema_documents.append(Document(text=table_schema))
    return full_schema, schema_documents, conn

# ------------------------------
# Function: Generate SQL Query using a PromptTemplate
# ------------------------------
def generate_sql_query(user_question, full_schema, query_engine, llm):
    """
    Constructs a prompt using developer instructions and an example,
    then appends the user's input. The prompt is wrapped in a PromptTemplate,
    so that predict() gets all required fields.
    """
    # Retrieve additional context (optional)
    retrieved_context = str(query_engine.query(user_question))
    
    developer_msg = (
        "You will be provided with a user query.\n"
        "Your goal is to generate a valid SQL query to provide the best answer to the user.\n\n"
        "This is the table schema:\n"
        f"{full_schema}\n\n"
        "Use this schema to generate as an output the SQL query.\n\n"
        "For example:\n\n"
        "USER INPUT: Show me the total revenue for each region for sales made in 2024.\n"
        "OUTPUT: SELECT Region, SUM(Quantity * Price) AS TotalRevenue\n"
        "FROM Sales\n"
        "WHERE Date BETWEEN '2024-01-01' AND '2024-12-31'\n"
        "GROUP BY Region;\n\n"
    )
    user_msg = f"USER INPUT: {user_question}"
    prompt_str = developer_msg + "Retrieved Context:\n" + retrieved_context + "\n\n" + user_msg

    # Create a PromptTemplate. The PromptTemplate constructor will set
    # default values for template_vars, metadata, kwargs, and output_parser.
    prompt = PromptTemplate(template=prompt_str)
    
    response = llm.predict(prompt)
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
def generate_analysis(sql_query, query_results, full_schema, llm):
    analysis_prompt_str = (
        f"Using the following SQLite database schema:\n{full_schema}\n\n"
        "Analyze the following SQL query and its results. Provide a brief explanation.\n\n"
        f"SQL Query: {sql_query}\n"
        f"Results: {query_results}\n"
        "Analysis:"
    )
    analysis_prompt = PromptTemplate(template=analysis_prompt_str)
    return llm.predict(analysis_prompt)

# ------------------------------
# Main Application
# ------------------------------
def main():
    # ------------------------------
    # Step 1: Set up OpenAI LLM in LlamaIndex Settings.
    # ------------------------------
    # Use the O3 model ("o3-mini") instead of GPT-4.
    Settings.llm = OpenAI(temperature=0.01, max_new_tokens=300, model="o3-mini")
    llm = Settings.llm  # We'll call llm.predict(prompt) below.

    # ------------------------------
    # Step 2: Load the SQLite database and extract its schema.
    # ------------------------------
    db_path = "./output/adventure_works.db"  # Ensure this file exists.
    full_schema, schema_docs, conn = load_db_schema(db_path)
    
    # ------------------------------
    # Step 3: Build a vector index over the schema documents.
    # ------------------------------
    embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
    index = VectorStoreIndex.from_documents(schema_docs, embed_model=embed_model)
    query_engine = index.as_query_engine()
    
    cursor = conn.cursor()
    print("SQLite database loaded successfully.\n")
    print("Database Schema:")
    print(full_schema)
    
    # ------------------------------
    # Step 4: Interactive Text-to-SQL Loop.
    # ------------------------------
    while True:
        user_question = input("Ask any question about the data (or type 'exit' to quit): ")
        if user_question.lower() in ["exit", "quit"]:
            break
        
        final_sql_query, full_response = generate_sql_query(user_question, full_schema, query_engine, llm)
        print("\nGenerated SQL Query:")
        print(final_sql_query)
        print("\nFull LLM Response:")
        print(full_response)
        
        try:
            cursor.execute(final_sql_query)
            query_results = cursor.fetchall()
        except Exception as e:
            query_results = f"Error executing query: {e}"
        print("\nQuery Results:")
        print(query_results)
        
        analysis = generate_analysis(final_sql_query, query_results, full_schema, llm)
        print("\nAnalysis:")
        print(analysis)
        print("\n" + "-" * 60 + "\n")
    
    conn.close()
    print("Application exited.")

if __name__ == "__main__":
    main()