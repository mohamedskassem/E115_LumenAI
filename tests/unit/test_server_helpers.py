import pytest
import os
import shutil
from unittest.mock import patch, MagicMock

# Functions to test
from src.app.server import cleanup_cache_dirs, shutdown_all_agents, load_global_data, create_new_chat_session
from src.app.server import UPLOAD_FOLDER, DB_FOLDER, SCHEMA_CACHE_DIR, DB_PATH, VECTOR_STORE_CACHE_DIR

# --- Tests for shutdown_all_agents --- #

def test_shutdown_all_agents_empty():
    """Test shutting down when no agents exist."""
    # Patch the global chat_agents dictionary for this test
    with patch.dict("src.app.server.chat_agents", {}, clear=True):
        shutdown_all_agents() # Should run without error
        # Assert the dictionary is still empty
        from src.app.server import chat_agents # Re-import to check state
        assert chat_agents == {}

def test_shutdown_all_agents_with_agents():
    """Test shutting down multiple agents."""
    mock_agent1 = MagicMock()
    mock_agent2 = MagicMock()
    initial_agents = {"agent1": mock_agent1, "agent2": mock_agent2}

    # Use patch.dict to modify the global in the server module
    with patch.dict("src.app.server.chat_agents", initial_agents, clear=True):
        shutdown_all_agents()

        # Assert shutdown was called on each mock agent
        mock_agent1.shutdown.assert_called_once()
        mock_agent2.shutdown.assert_called_once()

        # Assert the global dictionary is now empty
        from src.app.server import chat_agents # Re-import to check state
        assert chat_agents == {}

# --- Tests for cleanup_cache_dirs --- #

# Use mocker fixture for multiple patches
@pytest.fixture
def mock_cleanup_dependencies(mocker):
    """Mocks dependencies for cleanup_cache_dirs."""
    # Mock filesystem functions
    mock_exists = mocker.patch("src.app.server.os.path.exists")
    mock_rmtree = mocker.patch("src.app.server.shutil.rmtree")
    mock_remove = mocker.patch("src.app.server.os.remove")
    mock_makedirs = mocker.patch("src.app.server.os.makedirs")
    mock_sleep = mocker.patch("src.app.server.time.sleep") # Mock sleep to speed up tests

    # Mock schema cache path function
    mock_get_schema_path = mocker.patch("src.app.server._get_schema_cache_path", return_value="cache/schema.json")

    # Mock shutdown_all_agents to prevent side effects and check call
    mock_shutdown = mocker.patch("src.app.server.shutdown_all_agents")

    # Mock global state variables that are reset
    # We patch them directly in the test where needed usually,
    # but good to be aware they are modified.
    mocker.patch("src.app.server.is_data_loaded", True) # Start as True
    mocker.patch("src.app.server.shared_data_components", {"key": "value"})
    mocker.patch.dict("src.app.server.chat_agents", {"a": 1}, clear=True)

    return {
        "exists": mock_exists,
        "rmtree": mock_rmtree,
        "remove": mock_remove,
        "makedirs": mock_makedirs,
        "get_schema": mock_get_schema_path,
        "shutdown": mock_shutdown,
        "sleep": mock_sleep
    }

def test_cleanup_no_delete_db(mock_cleanup_dependencies):
    """Test cleanup when delete_db is False."""
    mocks = mock_cleanup_dependencies
    mocks["exists"].return_value = True # Assume upload folder exists

    errors, warnings = cleanup_cache_dirs(delete_db=False)

    assert errors == []
    assert warnings == []
    mocks["shutdown"].assert_called_once()
    mocks["exists"].assert_any_call(UPLOAD_FOLDER)
    mocks["rmtree"].assert_called_once_with(UPLOAD_FOLDER)
    mocks["remove"].assert_not_called() # DB and vector store not removed
    # Check makedirs is called to recreate essential dirs
    assert mocks["makedirs"].call_count == 3
    mocks["makedirs"].assert_any_call(UPLOAD_FOLDER, exist_ok=True)
    mocks["makedirs"].assert_any_call(DB_FOLDER, exist_ok=True)
    mocks["makedirs"].assert_any_call(SCHEMA_CACHE_DIR, exist_ok=True)
    # Check global state was NOT reset
    from src.app.server import is_data_loaded, shared_data_components, chat_agents
    assert is_data_loaded is True
    assert shared_data_components == {"key": "value"}
    # Note: chat_agents is cleared by shutdown_all_agents mock

def test_cleanup_delete_db_all_exist(mock_cleanup_dependencies):
    """Test cleanup when delete_db is True and all files/dirs exist."""
    mocks = mock_cleanup_dependencies
    mocks["exists"].return_value = True # Assume everything exists
    schema_cache_file = "cache/schema.json"
    mocks["get_schema"].return_value = schema_cache_file

    errors, warnings = cleanup_cache_dirs(delete_db=True)

    assert errors == []
    assert warnings == []
    mocks["shutdown"].assert_called_once()
    # Check existence checks
    mocks["exists"].assert_any_call(UPLOAD_FOLDER)
    mocks["exists"].assert_any_call(VECTOR_STORE_CACHE_DIR)
    mocks["exists"].assert_any_call(DB_PATH)
    mocks["exists"].assert_any_call(schema_cache_file)
    # Check removals
    mocks["rmtree"].assert_any_call(UPLOAD_FOLDER)
    mocks["rmtree"].assert_any_call(VECTOR_STORE_CACHE_DIR)
    mocks["remove"].assert_any_call(DB_PATH)
    mocks["remove"].assert_any_call(schema_cache_file)
    # Check recreation
    assert mocks["makedirs"].call_count == 3
    # Check global state WAS reset
    from src.app.server import is_data_loaded, shared_data_components, chat_agents
    assert is_data_loaded is False
    assert shared_data_components is None
    assert chat_agents == {}

def test_cleanup_delete_db_some_missing(mock_cleanup_dependencies):
    """Test cleanup when delete_db is True but some files/dirs are missing."""
    mocks = mock_cleanup_dependencies
    # Simulate only DB and schema cache existing
    mocks["exists"].side_effect = lambda p: p in [DB_PATH, "cache/schema.json"]
    schema_cache_file = "cache/schema.json"
    mocks["get_schema"].return_value = schema_cache_file

    errors, warnings = cleanup_cache_dirs(delete_db=True)

    assert errors == []
    assert warnings == []
    mocks["shutdown"].assert_called_once()
    # Check removals (only existing ones should be called)
    mocks["rmtree"].assert_not_called() # Upload/Vector cache don't exist
    mocks["remove"].assert_any_call(DB_PATH)
    mocks["remove"].assert_any_call(schema_cache_file)
    # Check recreation
    assert mocks["makedirs"].call_count == 3
    # Check global state WAS reset
    from src.app.server import is_data_loaded, shared_data_components, chat_agents
    assert is_data_loaded is False
    assert shared_data_components is None
    assert chat_agents == {}

def test_cleanup_rmtree_error(mock_cleanup_dependencies):
    """Test cleanup handles exceptions during rmtree."""
    mocks = mock_cleanup_dependencies
    mocks["exists"].return_value = True
    mocks["rmtree"].side_effect = OSError("Permission denied") # Simulate error

    errors, warnings = cleanup_cache_dirs(delete_db=True)

    assert len(errors) > 0
    assert "Permission denied" in errors[0]
    assert warnings == []
    mocks["shutdown"].assert_called_once()
    # Check global state WAS reset (even though cleanup had errors)
    from src.app.server import is_data_loaded, shared_data_components, chat_agents
    assert is_data_loaded is False
    assert shared_data_components is None
    assert chat_agents == {}

# --- Tests for load_global_data --- #

@patch("src.app.server.os.path.exists")
@patch("src.app.server.TextToSqlAgent.load_and_index_data")
def test_load_global_data_success(mock_load_agent_data, mock_exists):
    """Test successful global data loading."""
    mock_exists.return_value = True
    mock_loaded_data = {"key": "value"}
    mock_load_agent_data.return_value = mock_loaded_data

    # Need to patch globals directly for state checking
    with patch.dict("src.app.server.__dict__", {"is_data_loaded": False, "shared_data_components": None}):
        success = load_global_data(force_regenerate=False)

        assert success is True
        mock_exists.assert_called_once_with(DB_PATH)
        mock_load_agent_data.assert_called_once_with(
            db_path=DB_PATH,
            persist_dir=VECTOR_STORE_CACHE_DIR,
            force_regenerate_analysis=False
        )
        # Check global state was updated (need to import AFTER call)
        from src.app.server import is_data_loaded, shared_data_components
        assert is_data_loaded is True
        assert shared_data_components == mock_loaded_data

@patch("src.app.server.os.path.exists")
@patch("src.app.server.TextToSqlAgent.load_and_index_data")
def test_load_global_data_db_missing(mock_load_agent_data, mock_exists):
    """Test global data loading fails if DB file is missing."""
    mock_exists.return_value = False

    with patch.dict("src.app.server.__dict__", {"is_data_loaded": True, "shared_data_components": {}}):
        success = load_global_data()

        assert success is False
        mock_exists.assert_called_once_with(DB_PATH)
        mock_load_agent_data.assert_not_called()
        from src.app.server import is_data_loaded, shared_data_components
        assert is_data_loaded is False
        assert shared_data_components is None

@patch("src.app.server.os.path.exists")
@patch("src.app.server.TextToSqlAgent.load_and_index_data")
def test_load_global_data_agent_load_fails(mock_load_agent_data, mock_exists):
    """Test global data loading fails if agent loading returns None."""
    mock_exists.return_value = True
    mock_load_agent_data.return_value = None # Simulate failure

    with patch.dict("src.app.server.__dict__", {"is_data_loaded": True, "shared_data_components": {}}):
        success = load_global_data()

        assert success is False
        mock_exists.assert_called_once_with(DB_PATH)
        mock_load_agent_data.assert_called_once()
        from src.app.server import is_data_loaded, shared_data_components
        assert is_data_loaded is False
        assert shared_data_components is None


# --- Tests for create_new_chat_session --- #

@pytest.fixture
def mock_server_state(mocker):
    """Fixture to mock global state needed for create_new_chat_session."""
    # Mock the TextToSqlAgent class and its methods
    mock_agent_instance = MagicMock()
    mock_agent_instance.initialize_from_loaded_data.return_value = True
    mock_agent_class = mocker.patch("src.app.server.TextToSqlAgent", return_value=mock_agent_instance)

    # Mock UUID
    mock_uuid = mocker.patch("src.app.server.uuid.uuid4", return_value="mock-uuid-123")

    # Mock shared data components (assuming data is loaded)
    mock_shared_data = {
        "embed_model": MagicMock(),
        "full_schema": "mock schema",
        "detailed_schema_analysis": {"a": "b"},
        "query_engine": MagicMock()
    }
    mocker.patch("src.app.server.is_data_loaded", True)
    mocker.patch("src.app.server.shared_data_components", mock_shared_data)
    mocker.patch.dict("src.app.server.chat_agents", {}, clear=True)

    return {
        "agent_class": mock_agent_class,
        "agent_instance": mock_agent_instance,
        "uuid": mock_uuid,
        "shared_data": mock_shared_data
    }

def test_create_new_chat_session_success(mock_server_state):
    """Test successful creation of a new chat session."""
    mocks = mock_server_state

    new_chat_id = create_new_chat_session()

    assert new_chat_id == "mock-uuid-123"
    # Check agent was instantiated
    mocks["agent_class"].assert_called_once_with(db_path=DB_PATH, agent_id="mock-uuid-123")
    # Check agent was initialized
    mocks["agent_instance"].initialize_from_loaded_data.assert_called_once_with(
        embed_model=mocks["shared_data"]["embed_model"],
        full_schema=mocks["shared_data"]["full_schema"],
        detailed_schema_analysis=mocks["shared_data"]["detailed_schema_analysis"],
        query_engine=mocks["shared_data"]["query_engine"]
    )
    # Check agent was added to the global dict
    from src.app.server import chat_agents
    assert "mock-uuid-123" in chat_agents
    assert chat_agents["mock-uuid-123"] is mocks["agent_instance"]

def test_create_new_chat_session_data_not_loaded(mocker):
    """Test failure if global data is not loaded."""
    mocker.patch("src.app.server.is_data_loaded", False)
    mocker.patch("src.app.server.shared_data_components", None)
    mock_agent_class = mocker.patch("src.app.server.TextToSqlAgent")

    new_chat_id = create_new_chat_session()

    assert new_chat_id is None
    mock_agent_class.assert_not_called()

def test_create_new_chat_session_init_fails(mock_server_state):
    """Test failure if agent initialization fails."""
    mocks = mock_server_state
    # Simulate initialization failure
    mocks["agent_instance"].initialize_from_loaded_data.return_value = False

    new_chat_id = create_new_chat_session()

    assert new_chat_id is None
    # Check agent was instantiated
    mocks["agent_class"].assert_called_once()
    # Check initialization was attempted
    mocks["agent_instance"].initialize_from_loaded_data.assert_called_once()
    # Check shutdown was called on failed init
    mocks["agent_instance"].shutdown.assert_called_once()
    # Check agent was NOT added to the global dict
    from src.app.server import chat_agents
    assert "mock-uuid-123" not in chat_agents 