import logging
import sqlite3
from typing import Dict, Tuple, List, Union, Optional

# LlamaIndex components needed for type hints and prompt templating
from llama_index.core import PromptTemplate
from llama_index.llms.openai import OpenAI # Assuming OpenAI or compatible interface
from llama_index.llms.gemini import Gemini # For type hinting Union

# Define the type alias for the LLM instance
LlmType = Union[OpenAI, Gemini]

# --- Prompt Templates ---

VALIDATION_PROMPT_TEMPLATE = PromptTemplate(template="""
You are an AI assistant helping determine how to answer a user's question based on available database schema and conversation history.

Database Schema Summary:
{schema}

Conversation History:
{history}

User Question: "{question}"

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

OUTPUT:""")

SQL_GENERATION_PROMPT_TEMPLATE = PromptTemplate(template="""
You are an expert SQL generator. You will be provided with a user query, database schema, conversation history, and retrieved context.
Your goal is to generate a single, valid SQL query (compatible with SQLite) to provide the best answer to the user's most recent question.

Database Schema:
{schema}

Conversation History:
{history}

Retrieved Context (Information potentially relevant to the user query):
{context}

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

USER INPUT: {question}
GENERATED SQL:""")

ANALYSIS_PROMPT_TEMPLATE = PromptTemplate(template="""
You are LumenAI, a helpful data analyst AI. Your goal is to provide a clear, concise, and natural language response to the user's question based on the executed SQL query and its results. Incorporate context from the conversation history if relevant.

Conversation History:
{history}

Most Recent Interaction:
- User Question: {question}
- SQL Query Executed: {sql}
- Query Results:
{results}

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
""")

# --- LLM Task Functions ---

def validate_question(question: str, schema: str, history: str, llm: LlmType) -> Dict[str, str]:
    """Uses LLM to validate the user question against schema and history."""
    try:
        # Pass template and kwargs directly to predict
        response = llm.predict(
            VALIDATION_PROMPT_TEMPLATE, 
            schema=schema, 
            history=history, 
            question=question
        ).strip()
        logging.debug(f"Validation LLM response: {response}")

        parts = response.split(":", 1)
        action = parts[0].strip()
        details = parts[1].strip() if len(parts) > 1 else ""

        if action in ["SQL_NEEDED", "DIRECT_ANSWER", "CLARIFICATION_NEEDED"]:
            return {"action": action, "details": details}
        else:
            logging.warning(f"Unexpected validation response format: {response}. Defaulting to SQL_NEEDED.")
            return {"action": "SQL_NEEDED", "details": ""}
    except Exception as e:
        logging.error(f"Error during question validation LLM call: {e}", exc_info=True)
        # Fallback in case of API error
        return {"action": "SQL_NEEDED", "details": ""} # Default to SQL needed on error

def generate_sql(question: str, schema: str, history: str, context: str, llm: LlmType) -> Tuple[Optional[str], str]:
    """Generates SQL query using the LLM based on context, schema, and history."""
    try:
        # Pass template and kwargs directly to predict
        logging.info(f"Attempting SQL generation using LLM type: {type(llm)}")
        response = llm.predict(
            SQL_GENERATION_PROMPT_TEMPLATE, 
            schema=schema, 
            history=history, 
            context=context, 
            question=question
        ).strip()
        logging.info(f"Completed SQL generation using LLM type: {type(llm)}")
        logging.debug(f"SQL Generation LLM response: {response}")

        # Basic validation/cleanup of the generated SQL
        final_sql = response
        if final_sql.startswith("```sql"):
            final_sql = final_sql[len("```sql") :].strip()
        if final_sql.endswith("```"):
            final_sql = final_sql[: -len("```")].strip()
        final_sql = final_sql.strip().rstrip(";")

        logging.info(f"Generated SQL: {final_sql}")
        if not final_sql: # Handle case where LLM returns empty string
             logging.warning("SQL generation resulted in an empty string.")
             return None, "LLM returned an empty SQL query."
             
        return final_sql, response # Return cleaned SQL and raw response

    except Exception as e:
        logging.error(f"Error during SQL generation LLM call with {type(llm)}: {e}", exc_info=True)
        return None, f"Error generating SQL: {e}"

def generate_analysis(question: str, sql: str, query_results: Union[List[Tuple], str], history: str, llm: LlmType, cursor: Optional[sqlite3.Cursor]) -> str:
    """Generates natural language analysis of query results using LLM."""
    
    # Format results nicely for the prompt
    formatted_results: str
    if isinstance(query_results, str):  # Error message from execution
        formatted_results = query_results
    elif not query_results:
        formatted_results = "The query returned no results."
    else:
        # Convert results to a readable string, get headers from cursor
        headers = []
        if cursor and cursor.description:
            headers = [desc[0] for desc in cursor.description]
        
        results_str = "Headers: " + ", ".join(headers) + "\n" if headers else ""
        # Limit rows shown in prompt for brevity
        results_str += "\n".join([str(row) for row in query_results[:20]]) 
        if len(query_results) > 20:
            results_str += f"\n... (truncated, {len(query_results)} total rows)"
        formatted_results = results_str

    try:
        # Pass template and kwargs directly to predict
        analysis = llm.predict(
            ANALYSIS_PROMPT_TEMPLATE, 
            history=history, 
            question=question, 
            sql=sql, 
            results=formatted_results
        ).strip()
        logging.info("Analysis generated successfully.")
        logging.debug(f"Generated Analysis: {analysis}")
        return analysis
    except Exception as e:
        logging.error(f"Error during analysis generation LLM call: {e}", exc_info=True)
        return f"Error generating analysis: {e}" 