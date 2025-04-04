# LumenAI: Conversational Text-to-SQL Agent

This project implements a containerized conversational agent that allows users to query a database (e.g., Adventure Works) using natural language through a web interface.

## Key Features & Enhancements

*   **Web User Interface:** A simple, clean web UI (HTML/CSS/JS) for interacting with the agent, now rendering markdown in responses (using `marked.js`) for better readability.
*   **Flask Backend API:** A Flask server provides API endpoints to connect the frontend and the agent logic.
*   **OpenAPI Documentation:** Interactive API documentation available via Swagger UI (`/apidocs`), generated using Flasgger.
*   **Intelligent Query Validation:** The agent first validates user questions using an LLM to determine if SQL generation is needed (`SQL_NEEDED`), if the question can be answered directly (e.g., greetings, schema questions - `DIRECT_ANSWER`), or if clarification is required (`CLARIFICATION_NEEDED`).
*   **Contextual SQL Generation:** Generates SQL queries using schema information, retrieved context from a vector index, and conversation history.
*   **Multi-LLM Strategy:** 
    *   Uses `gpt-4-turbo` specifically for the intensive schema analysis during initialization or forced refresh (for potentially better accuracy).
    *   Uses a configurable model (defaulting to a Gemini model) for query-time validation, SQL generation, and results analysis.
*   **Vector-based Schema Understanding:** Leverages LlamaIndex embeddings (`text-embedding-3-small` by default) for semantic understanding of the database structure during context retrieval.
*   **Conversation History:** Maintains chat context across turns for improved follow-up queries and comparative analysis.
*   **Performance Optimization:** Implements schema analysis caching (`schema_cache.json`) and parallel processing during initial schema loading.
*   **Forced Index Refresh:** Provides an API endpoint (`/refresh-index`) to forcefully delete caches and rebuild the schema analysis and vector index.
*   **Containerized Deployment:** Fully containerized using Docker for easy setup and execution.
*   **Simplified Execution:** Includes a `dockershell.sh` script to build the Docker image and run the container with necessary volume mounts, environment variables (API keys), and port mappings.

## Architecture Overview

The application follows a three-tier architecture:

1.  **Frontend (Web UI):**
    *   Built with HTML, CSS, and vanilla JavaScript (`static/`, `templates/`).
    *   Includes `marked.js` to render markdown responses.
    *   Provides the chat interface for the user.
    *   Sends user questions to the backend API (`/query`).
    *   Displays responses and generated SQL from the backend.

2.  **Backend (Flask API):**
    *   Implemented in `server.py` using Flask and Flasgger.
    *   Exposes API endpoints (documented via `/apidocs`).
    *   Serves the static UI files (`GET /`).
    *   Manages the lifecycle of the `TextToSqlAgent`.
    *   Returns JSON responses to the frontend.

3.  **Agent Logic (`TextToSqlAgent`):**
    *   Contained within `app.py`.
    *   Manages conversation history.
    *   Performs initial question validation (SQL needed, direct answer, clarification) using the query-time LLM.
    *   Handles schema loading, analysis (using `gpt-4-turbo`), and caching.
    *   Uses LlamaIndex (`VectorStoreIndex`, `OpenAIEmbedding`) for semantic context retrieval.
    *   Interacts with LLMs (OpenAI `gpt-4-turbo` for schema, configured model like Gemini for queries) via LlamaIndex.
    *   Connects to the SQLite database (e.g., `output/user_data.db`) to execute generated queries.

```mermaid
graph LR
    A[Browser UI] -- HTTP Requests --> B(Flask API / Flasgger - server.py);
    B -- process_query() / manage agent --> C{TextToSqlAgent - app.py};
    subgraph Agent Initialization / Refresh
        C -- Load Schema / Analyze --> G[OpenAI GPT-4 Turbo];
    end
    subgraph Query Processing
         C -- "LLM Call (Validate/SQL/Analyze)" --> H[Configured LLM (e.g., Gemini)];
         C -- Retrieve Context --> E[Vector Index];
         C -- Execute SQL --> F[SQLite DB];
    end
    F -- Results --> C;
    G -- Analysis --> C;
    H -- Response --> C;
    E -- Context --> C;
    C -- Agent Response --> B;
    B -- HTTP Response (JSON / HTML) --> A;
    B -- Serves /apidocs --> A;
```

## Project Structure

```
.
├── README.md
├── LICENSE
├── .gitignore
└── src/
    └── app/                      # Main Application
        ├── Dockerfile
        ├── Pipfile
        ├── Pipfile.lock
        ├── .dockerignore
        ├── dockershell.sh        # Build & Run script
        ├── app.py                # Core TextToSqlAgent logic
        ├── server.py             # Flask server & API endpoints
        ├── schema_cache.json     # (Generated on first run)
        ├── secrets/
        │   ├── openai_api_key.txt # (User must create)
        │   └── google_api_key.json # (Optional, for Gemini ADC)
        ├── output/
        │   └── user_data.db      # Database file (e.g., from uploaded CSVs)
        ├── static/
        │   ├── script.js         # Frontend JavaScript
        │   └── style.css         # Frontend CSS
        ├── templates/
        │   └── index.html        # Frontend HTML
        └── vector_store_cache/   # (Generated on first run)
            └── ...
        # reports/ might exist containing project documentation
```

## API

The backend exposes API endpoints documented via Swagger UI.

*   **`GET /apidocs/`**: 
    *   **Purpose:** Serves the interactive Swagger UI documentation.
*   **`GET /`**: 
    *   **Purpose:** Serves the main HTML page (`index.html`) for the user interface.
*   **`GET /status`**: 
    *   **Purpose:** Checks if the agent is initialized and ready (based on DB presence).
*   **`POST /query`**: 
    *   **Purpose:** Receives a user question and returns the agent's response.
    *   **Request Body (JSON):** `{ "question": "<user_question>", "model": "<optional_llm_name>" }`
    *   **Response Body (JSON):** `{ "response_type": "<type>", "message": "<agent_response>", "sql_query": "<sql_or_null>" }` 
        *   `response_type` can be `sql_analysis`, `direct_answer`, `clarification_needed`, `sql_error`, or `error`.
*   **`POST /upload-data`**: 
    *   **Purpose:** Uploads one or more CSV files. Deletes existing data/caches, creates a new SQLite DB, and initializes the agent.
    *   **Request Body (multipart/form-data):** `files` field containing CSV file(s).
*   **`POST /remove-data`**: 
    *   **Purpose:** Shuts down the agent, deletes the database file, schema cache, and vector store cache.
*   **`POST /refresh-index`**: 
    *   **Purpose:** **Forces** a rebuild. Shuts down agent, deletes schema cache and vector store cache, then re-initializes the agent (re-analyzing schema with GPT-4 Turbo, rebuilding vector index) using the existing database file.

## Running Instructions

**Prerequisites:**

*   Docker installed and running.
*   Git (to clone the repository).
*   An OpenAI API key (with access to `gpt-4-turbo` for schema analysis).
*   (Optional) Google Cloud credentials correctly configured for Application Default Credentials (ADC) if using Gemini (e.g., via `gcloud auth application-default login` or a service account key file pointed to by `GOOGLE_APPLICATION_CREDENTIALS`).

**Steps:**

1.  **Clone the repository** (if you haven't already).
2.  **Navigate to the application directory:**
    ```bash
    cd path/to/repository/src/app 
    ```
3.  **Create Secrets File(s):**
    *   Create the directory `secrets` if it doesn't exist: `mkdir -p secrets`
    *   **OpenAI:** Create `secrets/openai_api_key.txt`. Paste *only* your OpenAI API key into this file and save it.
    *   **(Optional) Google:** If using Gemini and ADC via a service account file, place your `google_api_key.json` file in the `secrets` directory.
4.  **Run the Application:**
    *   Make the script executable (if necessary): `chmod +x dockershell.sh`
    *   Execute the script:
        ```bash
        bash dockershell.sh
        ```
        This script will:
        *   Build the Docker image (`text-to-sql-app`), installing dependencies including `flasgger`.
        *   Stop and remove any previously running container named `lumenai`.
        *   Run a new container named `lumenai`, mapping host port 5001 to container port 5000 (`-p 5001:5000`), passing the OpenAI API key environment variable, optionally setting `GOOGLE_APPLICATION_CREDENTIALS`, and mounting necessary volumes (`output`, `secrets`, `vector_store_cache`, `uploads`).

5.  **Access the Application:**
    *   Open your web browser and navigate to: `http://localhost:5001`
6.  **(First Run/No Data):** Use the UI to upload CSV files. This will create the `output/user_data.db`, analyze the schema, build the index, and enable the chat interface.
7.  **Access API Docs:** Navigate to `http://localhost:5001/apidocs/`

## Requirements Summary

*   Docker
*   Python 3.9 (as specified in Dockerfile)
*   OpenAI API Key (`gpt-4-turbo` access required)
*   (Optional) Google Cloud Credentials for Gemini
*   Required Python packages (installed via Pipfile within Docker): `Flask`, `flasgger`, `llama-index`, `llama-index-llms-openai`, `llama-index-llms-gemini`, `llama-index-embeddings-openai`, `openai`, `pandas`, etc.
