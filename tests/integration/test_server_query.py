import pytest
import json
from unittest.mock import patch, MagicMock

# Import the Flask app object
from src.app.server import app as flask_app

# Fixtures
@pytest.fixture(scope='module')
def app():
    flask_app.config.update({"TESTING": True})
    yield flask_app

@pytest.fixture(scope='module')
def client(app):
    return app.test_client()

# --- Tests for POST /query ---

@pytest.fixture
def mock_agents():
    """Provides a fixture for a mocked chat_agents dictionary."""
    mock_agent = MagicMock()
    # Give the mock agent a default title and process_query method
    mock_agent.chat_title = "Initial Title"
    mock_agent.is_initialized = True # Assume agent is ready
    # Configure process_query to return a default success response
    mock_agent.process_query.return_value = {
        "type": "sql_analysis",
        "message": "Here is the analysis.",
        "sql_query": "SELECT * FROM mock;"
    }
    # Mock the _get_llm method needed for title generation
    mock_agent._get_llm.return_value = MagicMock() # Return a mock LLM

    return {"test_chat_id": mock_agent}


def test_query_data_not_loaded(client):
    """Test query endpoint fails if data is not loaded."""
    with patch("src.app.server.is_data_loaded", False):
        response = client.post("/query", json={
            "question": "Test question?",
            "chat_id": "any_chat_id"
        })
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "Data not loaded" in data.get("error", "")

def test_query_missing_params(client):
    """Test query endpoint fails if chat_id or question is missing."""
    with patch("src.app.server.is_data_loaded", True):
        # Missing question
        response = client.post("/query", json={"chat_id": "test_chat_id"})
        assert response.status_code == 400
        assert "No question provided" in response.get_json().get("error", "")

        # Missing chat_id
        response = client.post("/query", json={"question": "Test question?"})
        assert response.status_code == 400
        assert "Missing chat_id" in response.get_json().get("error", "")

def test_query_chat_not_found(client):
    """Test query endpoint fails if chat_id doesn't exist."""
    with patch("src.app.server.is_data_loaded", True):
        # Patch chat_agents to be empty
        with patch("src.app.server.chat_agents", {}):
            response = client.post("/query", json={
                "question": "Test question?",
                "chat_id": "non_existent_chat"
            })
            assert response.status_code == 404
            assert "Chat session non_existent_chat not found" in response.get_json().get("error", "")

def test_query_agent_not_initialized(client, mock_agents):
    """Test query endpoint fails if the specific agent is not initialized."""
    # Modify the mock agent to be uninitialized
    mock_agents["test_chat_id"].is_initialized = False

    with patch("src.app.server.is_data_loaded", True):
        with patch("src.app.server.chat_agents", mock_agents):
            response = client.post("/query", json={
                "question": "Test question?",
                "chat_id": "test_chat_id"
            })
            assert response.status_code == 500 # Internal error state
            assert "Agent for chat test_chat_id is not ready" in response.get_json().get("error", "")
            # Reset agent state for other tests if needed (though fixtures usually handle this)
            mock_agents["test_chat_id"].is_initialized = True

def test_query_success(client, mock_agents):
    """Test a successful query call."""
    test_question = "Give me the results."
    expected_response_message = "Here is the analysis."
    expected_sql = "SELECT * FROM mock;"

    with patch("src.app.server.is_data_loaded", True):
        with patch("src.app.server.chat_agents", mock_agents):
            response = client.post("/query", json={
                "question": test_question,
                "chat_id": "test_chat_id"
            })
            data = response.get_json()

            assert response.status_code == 200
            assert data["response_type"] == "sql_analysis"
            assert data["message"] == expected_response_message
            assert data["sql_query"] == expected_sql
            assert data["chat_title"] == "Initial Title" # Check initial title is returned

            # Check that the agent's process_query was called correctly
            mock_agents["test_chat_id"].process_query.assert_called_once_with(
                test_question,
                "gemini-2.5-pro-preview-03-25" # Default model
            )

def test_query_agent_returns_error(client, mock_agents):
    """Test when the agent's process_query returns an error."""
    # Configure the mock agent to return an error response
    mock_agents["test_chat_id"].process_query.return_value = {
        "type": "error",
        "message": "Agent failed to process query.",
        "sql_query": "FAULTY SQL"
    }

    with patch("src.app.server.is_data_loaded", True):
        with patch("src.app.server.chat_agents", mock_agents):
            response = client.post("/query", json={
                "question": "This will fail",
                "chat_id": "test_chat_id"
            })
            data = response.get_json()

            assert response.status_code == 200 # Endpoint itself succeeds
            assert data["response_type"] == "error"
            assert data["message"] == "Agent failed to process query."
            assert data["sql_query"] == "FAULTY SQL"
            assert data["chat_title"] == "Initial Title"

def test_query_generates_title(client, mock_agents):
    """Test that a title is generated when requested for a new chat."""
    test_question = "First question generating title?"
    generated_title = "Generated Test Title"

    # Mock the title generation function itself
    with patch("src.app.server.generate_chat_title", return_value=generated_title) as mock_gen_title:
        # Mock the agent having the default title initially
        mock_agents["test_chat_id"].chat_title = "New Chat"

        with patch("src.app.server.is_data_loaded", True):
            with patch("src.app.server.chat_agents", mock_agents):
                response = client.post("/query", json={
                    "question": test_question,
                    "chat_id": "test_chat_id",
                    "generate_title": True # Explicitly request title generation
                })
                data = response.get_json()

                assert response.status_code == 200
                # Check title generation was called
                mock_gen_title.assert_called_once_with(test_question, mock_agents["test_chat_id"]._get_llm.return_value)
                # Check the agent's title was updated (in the mock)
                assert mock_agents["test_chat_id"].chat_title == generated_title
                # Check the response includes the *new* title
                assert data["chat_title"] == generated_title

def test_query_does_not_generate_title_if_not_requested(client, mock_agents):
    """Test title is not generated if generate_title is False."""
    with patch("src.app.server.generate_chat_title") as mock_gen_title:
        mock_agents["test_chat_id"].chat_title = "New Chat"
        with patch("src.app.server.is_data_loaded", True):
            with patch("src.app.server.chat_agents", mock_agents):
                response = client.post("/query", json={
                    "question": "Another question",
                    "chat_id": "test_chat_id",
                    "generate_title": False # Explicitly set to False
                })
                data = response.get_json()
                assert response.status_code == 200
                mock_gen_title.assert_not_called()
                assert data["chat_title"] == "New Chat" # Should retain default title

def test_query_does_not_generate_title_if_already_exists(client, mock_agents):
    """Test title is not generated if agent already has a non-default title."""
    with patch("src.app.server.generate_chat_title") as mock_gen_title:
        mock_agents["test_chat_id"].chat_title = "Existing Title"
        with patch("src.app.server.is_data_loaded", True):
            with patch("src.app.server.chat_agents", mock_agents):
                response = client.post("/query", json={
                    "question": "A question",
                    "chat_id": "test_chat_id",
                    "generate_title": True # Request generation, but it shouldn't happen
                })
                data = response.get_json()
                assert response.status_code == 200
                mock_gen_title.assert_not_called()
                assert data["chat_title"] == "Existing Title" # Should retain existing title 