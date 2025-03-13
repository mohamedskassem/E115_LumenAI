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
# Function: Load DB Schema and Column Summaries
# ------------------------------
def load_db_schema_and_column_summaries(db_path):
    """
    Loads the SQLite database schema and generates column statistics.

    Args:
        db_path (str): Path to the SQLite database file.

    Returns:
        tuple: (full_schema, schema_documents, conn)
            - full_schema (str): String representation of the database schema.
            - schema_documents (list): List of Document objects containing column statistics.
            - conn (sqlite3.Connection): Database connection object.
    """
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
            col_name = col[1]
            col_type = col[2]
            table_schema += f"  - {col_name} ({col_type})\n"

            # Generate column statistics using SQLite
            stats = {}
            try:
                if col_type.lower() in ("integer", "real", "numeric"):
                    cursor.execute(f"SELECT COUNT(*), MIN({col_name}), MAX({col_name}), AVG({col_name}) FROM {table_name} WHERE {col_name} IS NOT NULL")
                    count, min_val, max_val, avg_val = cursor.fetchone()
                    stats = {
                        "count": count,
                        "min": min_val,
                        "max": max_val,
                        "avg": avg_val,
                    }
                else:
                    cursor.execute(f"SELECT COUNT(DISTINCT {col_name}) FROM {table_name} WHERE {col_name} IS NOT NULL")
                    count_distinct = cursor.fetchone()[0]
                    stats = {
                        "distinct_count": count_distinct,
                    }
            except sqlite3.OperationalError as e:
                stats = {"error": f"Could not generate stats: {e}"}

            # Create document from statistics
            summary_text = f"Table: {table_name}, Column: {col_name}, Type: {col_type}, Stats: {stats}"
            schema_documents.append(Document(text=summary_text))

        table_schema += "\n"
        full_schema += table_schema

    return full_schema, schema_documents, conn

# ------------------------------
# Function: Generate SQL Query using a PromptTemplate
# ------------------------------
def generate_sql_query(user_question, full_schema, query_engine, llm):
    """
    Generates an SQL query based on the user's question and database schema.

    Args:
        user_question (str): The user's question.
        full_schema (str): String representation of the database schema.
        query_engine (llama_index.core.query_engine.BaseQueryEngine): Query engine for retrieving context.
        llm (llama_index.llms.llm.LLM): Language model for generating the query.

    Returns:
        tuple: (final_sql, response)
            - final_sql (str): Generated SQL query.
            - response (str): Full LLM response.
    """
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
def generate_analysis(sql_query, query_results, full_schema, llm, user_question):
    """
    Generates an analysis of the SQL query results in the context of the user's question.

    Args:
        sql_query (str): The generated SQL query.
        query_results (list): Results of executing the SQL query.
        full_schema (str): String representation of the database schema.
        llm (llama_index.llms.llm.LLM): Language model for generating the analysis.
        user_question (str): The original user question.

    Returns:
        str: Generated analysis.
    """
    analysis_prompt_str = (
        f"Using the following SQLite database schema:\n{full_schema}\n\n"
        "Analyze the following the query results in the context of the original user question. Provide a brief explanation.\n\n"
        f"Original User Question: {user_question}\n"
        f"SQL Query: {sql_query}\n"
        f"Results: {query_results}\n"
        "Analysis:"
    )
    analysis_prompt = PromptTemplate(template=analysis_prompt_str)
    return llm.predict(analysis_prompt)

import os
import sqlite3
from llama_index.core import VectorStoreIndex, Document, Settings, PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI

# ... (rest of the code: load_db_schema_and_column_summaries, generate_sql_query, generate_analysis)

# ------------------------------
# Main Application
# ------------------------------
def main():
    """
    Main application function.
    """
    Settings.llm = OpenAI(temperature=0.01, max_new_tokens=800, model="o3-mini")
    llm = Settings.llm

    db_path = "./output/adventure_works.db"
    full_schema, schema_docs, conn = load_db_schema_and_column_summaries(db_path)

    embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
    index = VectorStoreIndex.from_documents(schema_docs, embed_model=embed_model)
    query_engine = index.as_query_engine()

    cursor = conn.cursor()
    print("SQLite database loaded successfully.\n")
    print("Database Schema:")
    print(full_schema)

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

        analysis = generate_analysis(final_sql_query, query_results, full_schema, llm, user_question)
        print("\nAnalysis:")
        print(analysis)
        print("\n" + "-" * 60 + "\n")

    conn.close()
    print("Application exited.")

if __name__ == "__main__":
    main()