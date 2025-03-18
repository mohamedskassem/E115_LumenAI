import os
import sqlite3
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_path, device):
    """
    Loads the deepseek model and tokenizer from the local model folder.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float32  # Using float32 for CPU
    ).to(device)
    return tokenizer, model

def generate_text(tokenizer, model, prompt, device, max_new_tokens=512):
    """
    Uses the model to generate text from a prompt. This function ensures that 
    the input tensors are moved to the target device and handles both dictionary 
    and tensor outputs from the tokenizer.
    """
    messages = [{'role': 'user', 'content': prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        batch = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    else:
        batch = tokenizer(prompt, return_tensors="pt")
    
    # If the tokenizer output is a dict, move each tensor to the device.
    if isinstance(batch, dict):
        inputs = {key: value.to(device) for key, value in batch.items()}
        prompt_length = inputs["input_ids"].shape[1]
    else:
        inputs = batch.to(device)
        prompt_length = inputs.shape[1]
    
    # Generate output using the complete inputs.
    if isinstance(inputs, dict):
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id
        )
    else:
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id
        )
    
    generated = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
    return generated.strip()

def get_db_schema(conn):
    """
    Extracts the schema information (table names and columns) from the SQLite database.
    Returns the schema as a string.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    schema_info = ""
    for table_tuple in tables:
        table_name = table_tuple[0]
        schema_info += f"Table: {table_name}\n"
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        # PRAGMA table_info returns: cid, name, type, notnull, dflt_value, pk
        for col in columns:
            col_name = col[1]
            col_type = col[2]
            schema_info += f"  - {col_name} ({col_type})\n"
        schema_info += "\n"
    return schema_info

def basic_retrieve_context(user_query, schema_info):
    """
    A simple retrieval function that searches the schema for table names matching keywords
    from the user's question. If a match is found, the schema block for that table is returned.
    Otherwise, the full schema is returned.
    """
    lines = schema_info.split("\n")
    retrieved_context = ""
    i = 0
    found_any = False
    while i < len(lines):
        line = lines[i]
        if line.startswith("Table:"):
            table_name = line[len("Table:"):].strip()
            # Check if the table name appears in the user's query (case-insensitive).
            if table_name.lower() in user_query.lower():
                found_any = True
                block = []
                # Collect the table block (until an empty line).
                while i < len(lines) and lines[i].strip() != "":
                    block.append(lines[i])
                    i += 1
                retrieved_context += "\n".join(block) + "\n"
            else:
                i += 1
        else:
            i += 1
    # Fallback to the full schema if no specific context is retrieved.
    if not found_any:
        retrieved_context = schema_info
    return retrieved_context

def main():
    # Use CPU.
    device = "cpu"
    print(f"Using device: {device}")

    # Load the deepseek model from the local directory.
    model_path = './models/deepseek-coder-1.3b-instruct'
    tokenizer, model = load_model(model_path, device)
    
    # Connect to the SQLite database.
    db_path = './db/adventure_works.db'
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        print("SQLite database loaded successfully.")
    except Exception as e:
        print(f"Error loading database: {e}")
        return

    # Generate database schema information.
    schema_info = get_db_schema(conn)
    print("Database schema:")
    print(schema_info)

    while True:
        user_input = input("Ask any question about the data (or type 'exit' to quit): ")
        if user_input.lower() in ['exit', 'quit']:
            break

        # Retrieve context using our basic retrieval function.
        retrieved_context = basic_retrieve_context(user_input, schema_info)

        # Build the prompt with retrieved context and chain-of-thought instructions.
        sql_prompt = (
            f"Using the following retrieved context from the database schema:\n{retrieved_context}\n\n"
            f"And the full SQLite database schema:\n{schema_info}\n\n"
            f"And the following question: {user_input}\n\n"
            "First, outline your reasoning steps to construct a valid SQLite query that returns the correct result. "
            "Ensure you use valid SQLite functions (for example, use STRFTIME for date extraction) where needed. "
            "After your reasoning, on a new line, output exactly one line that begins with 'FinalSQL:' "
            "followed immediately by the final SQL query with no additional text or spaces or empty lines.\n"
 
        )
        print("\nGenerating SQL query with basic RAG and chain-of-thought reasoning...")
        sql_response = generate_text(tokenizer, model, sql_prompt, device, max_new_tokens=256)
        print(f"Full Response:\n{sql_response}\n")
        
        # Extract the final SQL query using the "FinalSQL:" delimiter.
        final_sql_query = None
        for line in sql_response.splitlines():
            if line.strip().startswith("FinalSQL:"):
                final_sql_query = line.strip()[len("FinalSQL:"):].strip()
                break
        if final_sql_query is None:
            # Fallback: use the last line of the response if no delimiter is found.
            final_sql_query = sql_response.splitlines()[-1].strip()

        print(f"Extracted SQL Query:\n{final_sql_query}\n")

        # Execute the generated SQL query on the SQLite database.
        try:
            cursor.execute(final_sql_query)
            query_results = cursor.fetchall()
        except Exception as e:
            query_results = f"Error executing query: {e}"
        print("Query Results:")
        print(query_results)

        # Generate an analysis of the query and its results.
        analysis_prompt = (
            f"Using the following SQLite database schema:\n{schema_info}\n\n"
            "Analyze the following SQL query and its results. Provide a brief explanation.\n\n"
            f"SQL Query: {final_sql_query}\n"
            f"Results: {query_results}\n"
            "Analysis:"
        )
        print("\nGenerating analysis...")
        analysis = generate_text(tokenizer, model, analysis_prompt, device, max_new_tokens=256)
        print("\nAnalysis:")
        print(analysis)
        print("\n" + "-"*60 + "\n")

    # Close the database connection.
    conn.close()
    print("Application exited.")

if __name__ == "__main__":
    main()
