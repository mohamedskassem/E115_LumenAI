import pytest
import sqlite3
import json
import os
from unittest.mock import patch, mock_open, MagicMock

# Functions/Constants to test
from src.app.data_handler import (
    get_db_connection,
    load_db_schema_and_analysis,
    _get_schema_cache_path,
    SCHEMA_ANALYSIS_CACHE_DIR,
    _process_table,
    _analyze_column,
    OpenAI,
    genai
)
# Import classes for spec mocking
from llama_index.core import Document

# --- Tests for get_db_connection --- #

@patch("src.app.data_handler.sqlite3.connect")
def test_get_db_connection_success(mock_connect):
    """Test successful database connection."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_connect.return_value = mock_conn
    db_path = "test_success.db"

    conn, cursor = get_db_connection(db_path)

    assert conn is mock_conn
    assert cursor is mock_cursor
    mock_connect.assert_called_once_with(db_path, check_same_thread=False)
    mock_conn.cursor.assert_called_once()

@patch("src.app.data_handler.sqlite3.connect")
def test_get_db_connection_failure(mock_connect):
    """Test database connection failure."""
    db_path = "test_fail.db"
    mock_connect.side_effect = sqlite3.Error("Connection failed")

    result = get_db_connection(db_path)

    assert result is None
    mock_connect.assert_called_once_with(db_path, check_same_thread=False)


# --- Tests for load_db_schema_and_analysis (Cache Logic) --- #

# Fixture to ensure cache dir exists for tests
@pytest.fixture(autouse=True)
def ensure_cache_dir_exists(tmp_path):
    # Use pytest's tmp_path fixture for a temporary directory
    cache_dir = tmp_path / SCHEMA_ANALYSIS_CACHE_DIR
    cache_dir.mkdir()
    # Patch the constant temporarily to use the tmp_path
    with patch("src.app.data_handler.SCHEMA_ANALYSIS_CACHE_DIR", str(cache_dir)):
        yield str(cache_dir) # Provide the temp cache dir path to tests if needed
    # tmp_path fixture handles cleanup automatically

@patch("src.app.data_handler.os.path.exists")
@patch("src.app.data_handler.open", new_callable=mock_open)
@patch("src.app.data_handler.json.load")
def test_load_from_cache_success(mock_json_load, mock_file_open, mock_os_exists):
    """Test successfully loading schema and analysis from a valid cache file."""
    db_path = "cached_db.db"
    cache_file_path = _get_schema_cache_path(db_path)

    mock_os_exists.return_value = True # Cache file exists

    # Define the mock cache data
    mock_cache_content = {
        "raw_schema": "CACHED SCHEMA",
        "analysis_results": [
            {"table": "t1", "column": "c1", "analysis": "a1", "text": "doc1"},
            {"table": "t1", "column": "c2", "analysis": "a2", "text": "doc2"}
        ]
    }
    mock_json_load.return_value = mock_cache_content

    # Mock the analysis LLM (shouldn't be called if cache hit)
    mock_llm = MagicMock()

    # Call the function
    full_schema, schema_docs, detailed_analysis = load_db_schema_and_analysis(
        db_path, mock_llm, force_regenerate=False
    )

    # Assertions
    mock_os_exists.assert_called_once_with(cache_file_path)
    mock_file_open.assert_called_once_with(cache_file_path, "r")
    mock_json_load.assert_called_once()
    assert full_schema == "CACHED SCHEMA"
    assert isinstance(schema_docs, list)
    assert len(schema_docs) == 2
    assert isinstance(schema_docs[0], Document)
    assert schema_docs[0].text == "doc1"
    assert schema_docs[1].text == "doc2"
    assert detailed_analysis == {"t1.c1": "a1", "t1.c2": "a2"}
    # Ensure DB connection wasn't attempted
    # (Need to mock connect *within* this test if we want to assert not called)
    with patch("src.app.data_handler.sqlite3.connect") as mock_connect:
        # Re-run logic conceptually (mocks already set up)
        load_db_schema_and_analysis(db_path, mock_llm, force_regenerate=False)
        mock_connect.assert_not_called()


@patch("src.app.data_handler.os.path.exists")
@patch("src.app.data_handler.sqlite3.connect") # Mock DB connection
def test_load_cache_missing(mock_connect, mock_os_exists):
    """Test cache miss (file doesn't exist) triggers DB load attempt."""
    db_path = "not_cached_db.db"
    cache_file_path = _get_schema_cache_path(db_path)

    mock_os_exists.return_value = False # Cache file does NOT exist

    # Mock the DB connection part to avoid errors down the line in this specific test
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.fetchall.return_value = [] # No tables found
    mock_connect.return_value = mock_conn

    # Call the function (we don't care about the actual return value here, just the flow)
    load_db_schema_and_analysis(db_path, MagicMock(), force_regenerate=False)

    # Assertions
    mock_os_exists.assert_called_once_with(cache_file_path)
    # Assert DB connection WAS attempted because cache was missed
    mock_connect.assert_called_once_with(db_path)


@patch("src.app.data_handler.os.path.exists")
@patch("src.app.data_handler.open", new_callable=mock_open)
@patch("src.app.data_handler.json.load")
@patch("src.app.data_handler.sqlite3.connect") # Mock DB connection
def test_load_cache_invalid_json(mock_connect, mock_json_load, mock_file_open, mock_os_exists):
    """Test invalid JSON in cache file triggers DB load attempt."""
    db_path = "invalid_cache.db"
    cache_file_path = _get_schema_cache_path(db_path)

    mock_os_exists.return_value = True # Cache file exists
    # Simulate JSON decode error
    mock_json_load.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)

    # Mock the DB connection part
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.fetchall.return_value = []
    mock_connect.return_value = mock_conn

    load_db_schema_and_analysis(db_path, MagicMock(), force_regenerate=False)

    # Assertions
    mock_os_exists.assert_called_once_with(cache_file_path)
    mock_file_open.assert_called_once_with(cache_file_path, "r")
    mock_json_load.assert_called_once()
    # Assert DB connection WAS attempted
    mock_connect.assert_called_once_with(db_path)


@patch("src.app.data_handler.os.path.exists")
@patch("src.app.data_handler.sqlite3.connect") # Mock DB connection
def test_load_force_regenerate(mock_connect, mock_os_exists):
    """Test force_regenerate=True skips cache check and loads from DB."""
    db_path = "force_regen.db"
    cache_file_path = _get_schema_cache_path(db_path)

    # Mock the DB connection part
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.fetchall.return_value = []
    mock_connect.return_value = mock_conn

    load_db_schema_and_analysis(db_path, MagicMock(), force_regenerate=True)

    # Assertions
    # Check that os.path.exists was NOT called for the cache file
    for call_args in mock_os_exists.call_args_list:
        assert call_args[0][0] != cache_file_path
    # Assert DB connection WAS attempted
    mock_connect.assert_called_once_with(db_path)

# Add more tests here for the main logic of load_db_schema_and_analysis (when generating fresh)
# and for _analyze_column / _process_table 

# --- Tests for load_db_schema_and_analysis (DB/Analysis Logic) --- #

# Mock data for DB interactions
@pytest.fixture
def mock_db_data():
    tables = [("users",), ("orders",)]
    users_info = [(0, 'id', 'INTEGER', 0, None, 1), (1, 'name', 'TEXT', 0, None, 0)]
    users_sample = [(1, 'Alice'), (2, 'Bob')]
    orders_info = [(0, 'order_id', 'INTEGER', 0, None, 1), (1, 'amount', 'REAL', 0, None, 0)]
    orders_sample = [(101, 50.5), (102, 75.0)]
    return {
        "tables": tables,
        "users_info": users_info,
        "users_sample": users_sample,
        "orders_info": orders_info,
        "orders_sample": orders_sample
    }

@patch("src.app.data_handler.os.path.exists", return_value=False) # Ensure cache miss
@patch("src.app.data_handler.sqlite3.connect")
@patch("src.app.data_handler._process_table") # Mock the helper processing function
@patch("src.app.data_handler.os.makedirs") # Mock makedirs for cache saving
@patch("src.app.data_handler.open", new_callable=mock_open)
@patch("src.app.data_handler.json.dump")
def test_load_db_schema_and_analysis_no_cache_success(
    mock_json_dump, mock_file_open, mock_makedirs, mock_process_table,
    mock_connect, mock_os_exists, mock_db_data
):
    """Test loading schema from DB when cache is missed, mocking _process_table."""
    db_path = "live_db.db"
    cache_file_path = _get_schema_cache_path(db_path)

    # --- Mock DB Setup ---
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_connect.return_value = mock_conn
    mock_conn.cursor.return_value = mock_cursor

    # Configure cursor fetchall for different queries
    def cursor_execute_side_effect(query):
        if "sqlite_master" in query:
            mock_cursor.fetchall.return_value = mock_db_data["tables"]
        elif 'PRAGMA table_info("users")' in query:
            mock_cursor.fetchall.return_value = mock_db_data["users_info"]
        elif 'PRAGMA table_info("orders")' in query:
            mock_cursor.fetchall.return_value = mock_db_data["orders_info"]
        elif 'SELECT * FROM "users"' in query:
            mock_cursor.fetchall.return_value = mock_db_data["users_sample"]
        elif 'SELECT * FROM "orders"' in query:
            mock_cursor.fetchall.return_value = mock_db_data["orders_sample"]
        else:
            mock_cursor.fetchall.return_value = [] # Default for unexpected calls
        return mock_cursor # Return self for execute

    mock_cursor.execute.side_effect = cursor_execute_side_effect

    # --- Mock _process_table --- #
    # Simulate _process_table returning schema string and analysis results
    process_results = [
        # Result for 'users' table
        ("Table: users\n  - id (INTEGER)\n  - name (TEXT)\n\n",
         [{"text": "udoc1", "table": "users", "column": "id", "analysis": "uid"}, {"text": "udoc2", "table": "users", "column": "name", "analysis": "uname"}]),
        # Result for 'orders' table
        ("Table: orders\n  - order_id (INTEGER)\n  - amount (REAL)\n\n",
         [{"text": "odoc1", "table": "orders", "column": "order_id", "analysis": "oid"}, {"text": "odoc2", "table": "orders", "column": "amount", "analysis": "oamount"}])
    ]
    mock_process_table.side_effect = process_results

    # --- Call Function --- #
    mock_llm = MagicMock() # LLM passed to _process_table
    full_schema, schema_docs, detailed_analysis = load_db_schema_and_analysis(
        db_path, mock_llm, force_regenerate=False
    )

    # --- Assertions --- #
    mock_os_exists.assert_called_once_with(cache_file_path)
    mock_connect.assert_called_once_with(db_path)
    assert mock_cursor.execute.call_count == 5 # Called for master, info/select per table

    # Check _process_table was called for each table
    assert mock_process_table.call_count == len(mock_db_data["tables"])
    # Check args of first call (users table)
    call_args_users, _ = mock_process_table.call_args_list[0]
    assert call_args_users[0][0] == "users" # table_name
    assert call_args_users[1] is mock_llm # llm instance
    # Check args of second call (orders table)
    call_args_orders, _ = mock_process_table.call_args_list[1]
    assert call_args_orders[0][0] == "orders" # table_name

    # Check returned values
    expected_full_schema = process_results[0][0] + process_results[1][0]
    assert full_schema == expected_full_schema
    assert len(schema_docs) == 4 # 2 docs per table
    assert schema_docs[0].text == "udoc1"
    assert schema_docs[3].text == "odoc2"
    assert detailed_analysis == {
        "users.id": "uid", "users.name": "uname",
        "orders.order_id": "oid", "orders.amount": "oamount"
    }

    # Check cache saving
    mock_makedirs.assert_called_once_with(os.path.dirname(cache_file_path), exist_ok=True)
    mock_file_open.assert_called_once_with(cache_file_path, "w")
    mock_json_dump.assert_called_once()
    # Check the structure dumped matches expectations
    dumped_data = mock_json_dump.call_args[0][0]
    assert dumped_data["raw_schema"] == expected_full_schema
    assert len(dumped_data["analysis_results"]) == 4
    assert dumped_data["analysis_results"][0]["text"] == "udoc1"
    assert dumped_data["analysis_results"][3]["analysis"] == "oamount"


# --- Tests for _analyze_column and _process_table --- #

@patch("src.app.data_handler._analyze_column")
def test_process_table(mock_analyze_column, mock_db_data):
    """Test _process_table processes columns and calls _analyze_column."""
    # Prepare input for a single table (users)
    table_name = "users"
    columns = mock_db_data["users_info"]
    sample_data = mock_db_data["users_sample"]
    column_indices = {col[1]: idx for idx, col in enumerate(columns)}
    table_info = (table_name, columns, sample_data, column_indices)
    mock_llm = MagicMock() # Mock LLM passed down

    # Define what _analyze_column should return for each column
    analyze_results = [
        {"text": "udoc1", "table": "users", "column": "id", "analysis": "uid"},
        {"text": "udoc2", "table": "users", "column": "name", "analysis": "uname"}
    ]
    mock_analyze_column.side_effect = analyze_results

    # Call the function
    table_schema, analysis_results_list = _process_table(table_info, mock_llm)

    # Assertions
    expected_schema_str = "Table: users\n  - id (INTEGER)\n  - name (TEXT)\n\n"
    assert table_schema == expected_schema_str
    assert analysis_results_list == analyze_results
    assert mock_analyze_column.call_count == len(columns)
    # Check call args for the first column ('id')
    mock_analyze_column.assert_any_call(
        'id', table_name, [1, 2], mock_llm # Sample data for 'id' column
    )
    # Check call args for the second column ('name')
    mock_analyze_column.assert_any_call(
        'name', table_name, ['Alice', 'Bob'], mock_llm # Sample data for 'name' column
    )

# Need OpenAI and GenerativeModel classes for spec
from src.app.data_handler import OpenAI, genai

@pytest.mark.parametrize(
    "llm_type, expected_call_method, mock_response, expected_summary",
    [
        ("openai", "predict", "OpenAI analysis result", "OpenAI analysis result"),
        ("gemini", "generate_content", MagicMock(text="Gemini analysis result", parts=[True]), "Gemini analysis result"), # Simulate successful Gemini response
        ("gemini_blocked", "generate_content", MagicMock(text=None, parts=[], prompt_feedback="Blocked"), "(LLM analysis blocked or empty)"), # Simulate blocked Gemini response
        ("none", None, None, "(LLM analysis not available)"), # No LLM provided
    ]
)
# Patch the PromptTemplate used by OpenAI predict call
@patch("src.app.data_handler.PromptTemplate")
def test_analyze_column(mock_prompt_template, llm_type, expected_call_method, mock_response, expected_summary):
    """Test _analyze_column with different LLM types and responses."""
    col_name = "description"
    table_name = "products"
    sample_data = ["long text 1", "long text 2", "short 3"]

    # Create mock LLM based on type
    mock_llm = None
    if llm_type == "openai":
        mock_llm = MagicMock(spec=OpenAI)
        mock_llm.predict.return_value = mock_response
    elif llm_type.startswith("gemini"):
        mock_llm = MagicMock(spec=genai.GenerativeModel)
        mock_llm.generate_content.return_value = mock_response
    # else llm_type == "none", mock_llm stays None

    # Mock the PromptTemplate construction (return self to allow chaining if needed)
    mock_prompt_template.return_value = mock_prompt_template

    # Call the function
    result = _analyze_column(col_name, table_name, sample_data, mock_llm)

    # Assertions
    if llm_type == "none":
        # Expecting a simple string back
        assert isinstance(result, str)
        assert f"Table: {table_name}" in result
        assert f"Column: {col_name}" in result
        assert expected_summary in result
    else:
        # Expecting a dictionary back
        assert isinstance(result, dict)
        assert result["table"] == table_name
        assert result["column"] == col_name
        assert result["analysis"] == expected_summary
        assert expected_summary in result["text"]

    # Check if the correct LLM method was called (or none)
    if expected_call_method == "predict":
        mock_llm.predict.assert_called_once()
        # Check that PromptTemplate was instantiated (called)
        mock_prompt_template.assert_called_once()
    elif expected_call_method == "generate_content":
        mock_llm.generate_content.assert_called_once()
        # PromptTemplate should NOT be called for Gemini
        mock_prompt_template.assert_not_called()
    else: # No LLM or failed LLM
        if mock_llm:
            # Check neither method was called if LLM exists but type is wrong (though test cases avoid this)
            mock_llm.predict.assert_not_called()
            mock_llm.generate_content.assert_not_called()

def test_analyze_column_llm_exception():
    """Test _analyze_column when LLM call raises an exception."""
    col_name = "error_col"
    table_name = "errors"
    sample_data = [1, 2]

    # Mock an OpenAI LLM that raises an error
    mock_llm = MagicMock(spec=OpenAI)
    mock_llm.predict.side_effect = Exception("API Rate Limit")

    # Patch PromptTemplate
    with patch("src.app.data_handler.PromptTemplate") as mock_pt:
        result = _analyze_column(col_name, table_name, sample_data, mock_llm)

        # Assertions
        assert result["analysis"] == "(Analysis failed due to error)"
        assert "Analysis failed due to error" in result["text"]
        mock_llm.predict.assert_called_once() # Ensure it was called



# Add more tests here for the main logic of load_db_schema_and_analysis (when generating fresh)
# and for _analyze_column / _process_table