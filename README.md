# LumenAI: Conversational Text-to-SQL Agent

![LumenAI User Interface](img/UI.png)

This project implements a containerized conversational agent that allows users to query a database (e.g., Adventure Works) using natural language through a web interface.

## Key Features & Enhancements

*   **Multi-Chat Web Interface:** A simple, clean web UI (HTML/CSS/JS) featuring a **sidebar for managing multiple independent chat sessions**. Users can create new chats, switch between them, and delete individual chats via an icon in the sidebar.
*   **Per-Chat Conversation History:** Maintains chat context **within each individual chat session** for improved follow-up queries and comparative analysis. User and bot messages are correctly reloaded when switching chats.
*   **Automatic Chat Titles:** New chats are automatically titled based on the content of the first user query, making sessions easier to identify.
*   **Inline SQL Toggle:** Generated SQL queries are displayed within an expandable toggle inside the relevant bot message bubble, rather than a fixed area.
*   **Configurable LLM:** Choose the query LLM (e.g., GPT-4 Turbo, Gemini 2.5 Pro) via a dropdown in the chat header.
*   **Flask Backend API:** A Flask server provides API endpoints to connect the frontend and the agent logic.
*   **OpenAPI Documentation:** Interactive API documentation available via Swagger UI (`/apidocs`), generated using Flasgger.
*   **Intelligent Query Validation:** The agent first validates user questions using an LLM to determine if SQL generation is needed (`SQL_NEEDED`), if the question can be answered directly (e.g., greetings, schema questions - `DIRECT_ANSWER`), or if clarification is required (`CLARIFICATION_NEEDED`), using the conversation history *specific to the current chat*.
*   **Contextual SQL Generation:** Generates SQL queries using schema information, retrieved context from a vector index, **enhanced schema analysis**, and the conversation history *specific to the current chat*.
*   **SQL Execution Retry Logic:** If a generated SQL query fails execution, the agent automatically analyzes the error, passes it back to the LLM to generate a corrected query, and retries the execution (up to 3 attempts).
*   **Multi-LLM Strategy:**
    *   Uses `gpt-4-turbo` specifically for the intensive **schema analysis** during initialization or forced refresh (for potentially better accuracy).
    *   Uses a configurable model (defaulting to a Gemini model) for query-time validation, SQL generation, results analysis, and chat title generation.
*   **Vector-based Schema Understanding:** Leverages LlamaIndex embeddings (`text-embedding-3-small` by default) for semantic understanding of the database structure during context retrieval.
*   **Cached Schema Analysis:**
    *   On first load (or refresh) for a database, the agent uses an LLM (`gpt-4-turbo`) to generate detailed descriptions/analysis for each table and column.
    *   This analysis is **cached** to a JSON file (e.g., `schema_analysis_cache/user_data.db.json`) specific to the database being loaded.
    *   On subsequent startups, the agent **loads the analysis from the cache**, significantly speeding up initialization and reducing LLM calls.
    *   The cache file is automatically deleted if the corresponding database is removed via the UI.
*   **Enhanced LLM Context:** The **cached detailed schema analysis** is provided as additional context to the LLM during question validation, SQL generation, and results analysis, improving accuracy and understanding.
*   **Containerized Deployment:** Fully containerized using Docker for easy setup and execution.
*   **Simplified Execution:** Includes a `dockershell.sh` script to build the Docker image and run the container with necessary volume mounts (including persistence for the schema analysis cache), environment variables (API keys), and port mappings.

## Architecture Overview

The application follows a three-tier architecture:

1.  **Frontend (Web UI):**
    *   Built with HTML, CSS, and vanilla JavaScript (`static/`, `templates/`).
    *   Includes `marked.js` to render markdown responses.
    *   Provides the multi-chat interface (sidebar, chat area) for the user.
    *   Sends user questions and chat management requests to the backend API.
    *   Displays responses and inline SQL toggles from the backend.

2.  **Backend (Flask API):**
    *   Implemented in `server.py` using Flask and Flasgger.
    *   Exposes API endpoints (documented via `/apidocs`).
    *   Serves the static UI files (`GET /`).
    *   Manages the overall application state:
        *   Holds **shared, pre-loaded data components** (embedding model, schema analysis, query engine) once data is uploaded (`shared_data_components`).
        *   Manages a dictionary of active `TextToSqlAgent` instances, keyed by `chat_id` (`chat_agents`). Each agent instance uses the shared components but maintains its own conversation history and state.
    *   Routes incoming requests (e.g., `/query`, `/chats/<chat_id>`) to the appropriate agent instance based on the `chat_id`.
    *   Handles data upload, deletion, and initial chat creation.
    *   Returns JSON responses to the frontend.

3.  **Agent Logic (`TextToSqlAgent` & Modules):**
    *   **`app.py` (`TextToSqlAgent`):** Represents a single chat session. Orchestrates the query process (including validation, SQL generation/retry loop, analysis), manages **per-agent conversation history** (`ConversationHistory`), stores the `chat_title`, handles LLM instance caching for the agent (`_get_llm`), and coordinates interactions with other modules using the shared data components.
    *   **`data_handler.py`:** Responsible for connecting to the database, generating/caching/loading the shared detailed **LLM schema analysis**, providing the raw schema string, and establishing DB connections (used per-agent).
    *   **`llm_interface.py`:** Contains the specific prompt templates and functions for interacting with LLMs (using raw schema, detailed analysis context, and per-chat history) for core tasks: question validation, SQL generation, results analysis, and **chat title generation**.
    *   **`vector_store.py` (`VectorStoreManager`):** Manages the lifecycle of the shared LlamaIndex `VectorStoreIndex` - checking cache validity, loading from cache, building a new index, persisting, and providing the shared query engine.
    *   The agent uses LlamaIndex (`VectorStoreIndex`, `OpenAIEmbedding`) for semantic context retrieval (via the shared `VectorStoreManager`).
    *   Interacts with LLMs (OpenAI `gpt-4-turbo` for schema analysis via `data_handler`, configured model like Gemini for query/title tasks via `llm_interface`).

```mermaid
graph LR
    A[Browser UI] -- HTTP Requests (incl. chat_id) --> B(Flask API / Flasgger - server.py);

    subgraph Backend State Management
        B -- Manages --> AGENTS{Active Agents (Dict[chat_id, TextToSqlAgent])};
        B -- Manages --> SHARED[Shared Data Components (Index, Embeddings, Schema Analysis)];
    end

    B -- Route request based on chat_id --> C(Specific TextToSqlAgent Instance - app.py);

    subgraph Agent Core Logic
        C -- Uses --> SHARED;
        C -- Manages --> HIST[Per-Agent History];
        C -- Needs LLM Task --> I[LLM Interface - llm_interface.py];
        C -- Execute SQL --> F[(SQLite DB)];
    end

    subgraph Shared Components Initialization (on data load)
        D[Data Handler - data_handler.py] -- Generates/Loads --> SHARED;
        J[Vector Store Mgr - vector_store.py] -- Builds/Loads --> SHARED;
        D -- Uses --> G[OpenAI GPT-4 Turbo for Analysis];
        J -- Uses --> K[Embedding Model];
        D -- Accesses --> F;
        J -- Accesses --> E[Vector Index Store];
        J -- Uses Schema Docs from --> D;
    end

    subgraph Query Processing
        I -- LLM Call (w/ Shared Analysis + Per-Agent History) --> H[Configured Query LLM];
        H -- Response --> I;
        I -- Result --> C;
        SHARED -- Query Engine --> C;
        F -- SQL Results / Errors --> C;
    end

    C -- Agent Response --> B;
    B -- HTTP Response (JSON / HTML) --> A;
    B -- Serves /apidocs --> A;
```
*(Diagram updated to reflect multi-agent architecture)*

### Architecture Diagram Images
*(Note: Linked images below might be outdated compared to the Mermaid diagram above)*
![Architecture Diagram 1](img/arch1.png)
![Architecture Diagram 2](img/arch2.png)
![Architecture Diagram 3](img/arch3.png)

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
        ├── app.py                # Core TextToSqlAgent logic (Orchestration per chat)
        ├── server.py             # Flask server & API endpoints (Multi-agent mgmt)
        ├── data_handler.py       # Database schema loading and analysis logic + Caching
        ├── llm_interface.py      # LLM prompt templates and task execution functions (incl. title)
        ├── vector_store.py       # Vector store index management logic
        ├── secrets/
        │   ├── openai_api_key.txt # (User must create)
        │   └── google_api_key.txt # (User must create, holds Google API Key)
        ├── uploads/              # (Generated) Temp storage for uploaded CSVs
        ├── output/               # (Generated) Database file(s) live here (e.g., user_data.db)
        ├── static/
        │   ├── script.js         # Frontend JavaScript (handles multi-chat UI)
        │   ├── style.css         # Frontend CSS
        │   ├── logo.svg
        │   ├── favicon.svg
        │   └── delete.svg        # Delete icon for chat list
        ├── templates/
        │   └── index.html        # Frontend HTML (with sidebar)
        ├── vector_store_cache/   # (Generated) Vector index cache
            └── ...
        └── schema_analysis_cache/ # (Generated) LLM Schema analysis cache
            └── user_data.db.json # Example cache file name
        # reports/ might exist containing project documentation
```

## API

The backend exposes API endpoints documented via Swagger UI (`/apidocs`).

*   **`GET /apidocs/`**:
    *   **Purpose:** Serves the interactive Swagger UI documentation.
*   **`GET /`**:
    *   **Purpose:** Serves the main HTML page (`index.html`) for the user interface.
*   **`GET /status`**:
    *   **Purpose:** Checks if data is loaded and returns the list of active chats.
    *   **Response Body (JSON):** `{ "is_data_loaded": <boolean>, "active_chats": [ { "chat_id": "<id>", "title": "<title>" }, ... ] }`
*   **`GET /chats`**:
    *   **Purpose:** Gets a list of active chat session IDs (Simple list).
    *   **Response Body (JSON):** `[ "<chat_id_1>", "<chat_id_2>", ... ]`
*   **`POST /chats`**:
    *   **Purpose:** Creates a new chat session instance. Requires data to be loaded first.
    *   **Response Body (JSON):** `{ "chat_id": "<new_chat_id>" }`
*   **`DELETE /chats/<chat_id>`**:
    *   **Purpose:** Deletes a specific chat session instance.
    *   **Response Body (JSON):** `{ "message": "Chat session <chat_id> deleted." }`
*   **`POST /query`**:
    *   **Purpose:** Receives a user question for a specific chat session and returns the agent's response. Handles SQL generation/execution with retries, and potentially generates a chat title on the first query.
    *   **Request Body (JSON):** `{ "question": "<user_question>", "chat_id": "<target_chat_id>", "model": "<optional_llm_name>", "generate_title": <boolean_optional> }`
    *   **Response Body (JSON):** `{ "response_type": "<type>", "message": "<agent_response>", "sql_query": "<sql_or_null>", "chat_title": "<current_chat_title>" }`
        *   `response_type` can be `sql_analysis`, `direct_answer`, `clarification_needed`, `error`.
*   **`POST /upload-data`**:
    *   **Purpose:** Uploads one or more CSV files. Deletes ALL existing data/caches/chats, creates a new SQLite DB (`output/user_data.db`), initializes shared components (schema analysis, indexing), and creates the *first* initial chat session.
    *   **Request Body (multipart/form-data):** `files` field containing CSV file(s).
    *   **Response Body (JSON):** `{ "message": "...", "new_chat_id": "<id_of_first_chat>" }`
*   **`POST /remove-data`**:
    *   **Purpose:** Shuts down all agent instances, deletes the database file (`output/user_data.db`), its corresponding schema analysis cache file (e.g., `schema_analysis_cache/user_data.db.json`), the vector store cache directory (`vector_store_cache/`), clears all chat states, and resets the UI to the data upload state.
    *   **Response Body (JSON):** `{ "message": "...", "warnings": [...] }` or `{ "message": "...", "errors": [...], "warnings": [...] }`

## Running Instructions

**Prerequisites:**

*   Docker installed and running.
*   Git (to clone the repository).
*   An OpenAI API key (with access to `gpt-4-turbo` for schema analysis).
*   A Google API Key (for the default query model, Gemini).

**Steps:**

1.  **Clone the repository** (if you haven't already).
2.  **Navigate to the application directory:**
    ```bash
    cd path/to/repository/src/app
    ```
3.  **Create Secrets File(s):**
    *   Create the directory `secrets` if it doesn't exist: `mkdir -p secrets`
    *   **OpenAI:** Create `secrets/openai_api_key.txt`. Paste *only* your OpenAI API key into this file and save it.
    *   **Google:** Create `secrets/google_api_key.txt`. Paste *only* your Google API key into this file and save it.
4.  **Run the Application:**
    *   Make the script executable (if necessary): `chmod +x dockershell.sh`
    *   Execute the script:
        ```bash
        bash dockershell.sh
        ```
        This script will:
        *   Build the Docker image (`text-to-sql-app`), installing dependencies including `flasgger`.
        *   Stop and remove any previously running container named `lumenai`.
        *   Run a new container named `lumenai`, mapping host port 5001 to container port 5000 (`-p 5001:5000`), passing the OpenAI and Google API key environment variables, and mounting necessary volumes (`output`, `secrets`, `vector_store_cache`, `schema_analysis_cache`, `uploads`).

5.  **Access the Application:**
    *   Open your web browser and navigate to: `http://localhost:5001`
6.  **(First Run/No Data):** Use the UI to upload CSV files. This will create the `output/user_data.db`, analyze the schema (using GPT-4 Turbo), save the analysis cache, build the vector index, create an initial chat session, and enable the chat interface.
7.  **Access API Docs:** Navigate to `http://localhost:5001/apidocs/`

## Requirements Summary

*   Docker
*   Python 3.9+ (as specified in Dockerfile)
*   OpenAI API Key (`gpt-4-turbo` access required for schema analysis)
*   Google API Key (required for default query/title model)
*   Required Python packages (installed via Pipfile within Docker): `Flask`, `flasgger`, `google-generativeai`, `llama-index`, `llama-index-llms-openai`, `llama-index-llms-gemini`, `llama-index-embeddings-openai`, `openai`, `pandas`, etc.
