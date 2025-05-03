import pytest
import json
from unittest.mock import patch, MagicMock

# Import the Flask app object and potentially objects to mock
from src.app.server import app as flask_app
# from src.app.app import TextToSqlAgent # Not strictly needed if fully mocking

# Fixtures from test_server_status or a shared conftest.py
# For simplicity, redefining client fixture here
@pytest.fixture(scope='module')
def app():
    flask_app.config.update({"TESTING": True})
    yield flask_app

@pytest.fixture(scope='module')
def client(app):
    return app.test_client()

# --- Tests for POST /chats ---

def test_create_chat_success(client):
    """Test successfully creating a new chat when data is loaded."""
    # Mock that data is loaded and create_new_chat_session returns a new ID
    with patch("src.app.server.is_data_loaded", True):
        with patch("src.app.server.create_new_chat_session", return_value="new_chat_id_123") as mock_create:
            response = client.post("/chats")
            assert response.status_code == 201
            data = json.loads(response.data)
            assert data["chat_id"] == "new_chat_id_123"
            mock_create.assert_called_once()

def test_create_chat_data_not_loaded(client):
    """Test creating a chat fails if data is not loaded."""
    with patch("src.app.server.is_data_loaded", False):
        with patch("src.app.server.create_new_chat_session") as mock_create:
            response = client.post("/chats")
            assert response.status_code == 400
            data = json.loads(response.data)
            assert "error" in data
            assert "Data not loaded" in data["error"]
            mock_create.assert_not_called()

def test_create_chat_creation_fails(client):
    """Test creating a chat fails if create_new_chat_session returns None."""
    with patch("src.app.server.is_data_loaded", True):
        # Simulate failure in the helper function
        with patch("src.app.server.create_new_chat_session", return_value=None) as mock_create:
            response = client.post("/chats")
            assert response.status_code == 500
            data = json.loads(response.data)
            assert "error" in data
            assert "Failed to create new chat session" in data["error"]
            mock_create.assert_called_once()

# --- Tests for GET /chats ---

def test_get_chats_empty(client):
    """Test getting chat list when none exist."""
    # Need to mock the global chat_agents dictionary
    with patch("src.app.server.chat_agents", {}):
        response = client.get("/chats")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert isinstance(data, list)
        assert len(data) == 0

def test_get_chats_with_data(client):
    """Test getting chat list with active chats."""
    # Mock the chat_agents dictionary with some dummy keys
    mock_agents = {"id1": MagicMock(), "id2": MagicMock(), "id3": MagicMock()}
    with patch("src.app.server.chat_agents", mock_agents):
        response = client.get("/chats")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert isinstance(data, list)
        # Order is not guaranteed by dict keys, so check length and presence
        assert len(data) == 3
        assert "id1" in data
        assert "id2" in data
        assert "id3" in data

# --- Tests for DELETE /chats/<chat_id> ---

def test_delete_chat_success(client):
    """Test successfully deleting an existing chat."""
    # Mock an agent that can be deleted
    mock_agent = MagicMock()
    chat_id_to_delete = "chat_to_delete_123"
    # Mock the dictionary *before* the request
    mock_chat_agents = {chat_id_to_delete: mock_agent}

    # Patch the dictionary that the route will modify
    with patch.dict("src.app.server.chat_agents", mock_chat_agents, clear=True):
        response = client.delete(f"/chats/{chat_id_to_delete}")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "message" in data
        assert chat_id_to_delete in data["message"]
        # Check the agent's shutdown was called
        mock_agent.shutdown.assert_called_once()

def test_delete_chat_not_found(client):
    """Test deleting a chat ID that doesn't exist."""
    chat_id_to_delete = "non_existent_chat_456"
    # Ensure the dictionary is empty or doesn't contain the ID
    with patch.dict("src.app.server.chat_agents", {}, clear=True):
        response = client.delete(f"/chats/{chat_id_to_delete}")
        assert response.status_code == 404
        data = json.loads(response.data)
        assert "error" in data
        assert chat_id_to_delete in data["error"] 