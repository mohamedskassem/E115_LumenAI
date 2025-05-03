import pytest
import json
from unittest.mock import patch, MagicMock

# Import the Flask app object
# Assuming your Flask app instance is named 'app' in src.app.server
from src.app.server import app as flask_app

# Configure the Flask app for testing
# This changes the environment to testing, disabling error catching
# during request handling, so you get better error reports
@pytest.fixture(scope='module')
def app():
    flask_app.config.update({
        "TESTING": True,
    })
    # You might add other test-specific configurations here
    yield flask_app

# Create a test client fixture using the app fixture
@pytest.fixture(scope='module')
def client(app):
    return app.test_client()

# Test the /status endpoint when data is NOT loaded
def test_status_not_loaded(client):
    """Test the /status endpoint when is_data_loaded is False."""
    # Mock the global variable `is_data_loaded` within the server module
    with patch("src.app.server.is_data_loaded", False):
        # Also mock chat_agents as empty
        with patch("src.app.server.chat_agents", {}):
            response = client.get("/status")
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data["is_data_loaded"] is False
            assert "active_chats" in data
            assert data["active_chats"] == []

# Test the /status endpoint when data IS loaded
def test_status_loaded_no_chats(client):
    """Test the /status endpoint when is_data_loaded is True but no chats exist."""
    # Mock the global variable `is_data_loaded`
    with patch("src.app.server.is_data_loaded", True):
        # Mock chat_agents as empty
        with patch("src.app.server.chat_agents", {}):
            response = client.get("/status")
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data["is_data_loaded"] is True
            assert data["active_chats"] == []

def test_status_loaded_with_chats(client):
    """Test the /status endpoint when is_data_loaded is True and chats exist."""
    # Create mock agents with chat_title attribute
    mock_agent1 = MagicMock()
    mock_agent1.chat_title = "Chat about Sales"
    mock_agent2 = MagicMock()
    mock_agent2.chat_title = "Customer Analysis"

    mock_chat_agents = {
        "chat123": mock_agent1,
        "chat456": mock_agent2
    }

    # Mock the global variables
    with patch("src.app.server.is_data_loaded", True):
        with patch("src.app.server.chat_agents", mock_chat_agents):
            response = client.get("/status")
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data["is_data_loaded"] is True
            assert len(data["active_chats"]) == 2
            # Check if the content is correct (order might not be guaranteed)
            expected_chats = [
                {"chat_id": "chat123", "title": "Chat about Sales"},
                {"chat_id": "chat456", "title": "Customer Analysis"}
            ]
            # Sort both lists by chat_id to ensure comparison is order-independent
            assert sorted(data["active_chats"], key=lambda x: x['chat_id']) == sorted(expected_chats, key=lambda x: x['chat_id']) 