import pytest
from src.app.app import ConversationHistory

# Fixture for ConversationHistory instance
@pytest.fixture
def history_manager():
    return ConversationHistory(max_history=3)

def test_history_initialization(history_manager):
    """Test if the history initializes correctly."""
    assert history_manager.history == []
    assert history_manager.max_history == 3

def test_add_single_interaction(history_manager):
    """Test adding a single interaction."""
    history_manager.add_interaction(
        question="What is the total sales?",
        response_type="sql_analysis",
        sql_query="SELECT SUM(sales) FROM orders;",
        results="[(1000,)]",
        analysis="Total sales are 1000."
    )
    assert len(history_manager.history) == 1
    interaction = history_manager.history[0]
    assert interaction["question"] == "What is the total sales?"
    assert interaction["response_type"] == "sql_analysis"
    assert interaction["sql_query"] == "SELECT SUM(sales) FROM orders;"
    assert interaction["results"] == "[(1000,)]"
    assert interaction["analysis"] == "Total sales are 1000."

def test_add_multiple_interactions_within_limit(history_manager):
    """Test adding multiple interactions within the max_history limit."""
    history_manager.add_interaction(question="Q1", response_type="direct_answer", analysis="A1")
    history_manager.add_interaction(question="Q2", response_type="sql_analysis", sql_query="SQL2", analysis="A2")
    assert len(history_manager.history) == 2
    assert history_manager.history[0]["question"] == "Q1"
    assert history_manager.history[1]["question"] == "Q2"

def test_add_interactions_exceeding_limit(history_manager):
    """Test that adding interactions beyond max_history removes the oldest one."""
    history_manager.add_interaction(question="Q1", response_type="direct_answer", analysis="A1")
    history_manager.add_interaction(question="Q2", response_type="sql_analysis", sql_query="SQL2", analysis="A2")
    history_manager.add_interaction(question="Q3", response_type="direct_answer", analysis="A3")
    history_manager.add_interaction(question="Q4", response_type="clarification_needed", analysis="A4") # This should push Q1 out

    assert len(history_manager.history) == 3 # Max history is 3
    assert history_manager.history[0]["question"] == "Q2" # Q1 should be gone
    assert history_manager.history[1]["question"] == "Q3"
    assert history_manager.history[2]["question"] == "Q4"

def test_get_formatted_history_empty(history_manager):
    """Test formatting an empty history."""
    assert history_manager.get_formatted_history() == ""

def test_get_formatted_history_single_interaction(history_manager):
    """Test formatting history with one complete interaction."""
    history_manager.add_interaction(
        question="What are the sales?",
        response_type="sql_analysis",
        sql_query="SELECT sales FROM data;",
        results="[(100,), (200,)]",
        analysis="Sales values are 100 and 200."
    )
    expected_format = (
        "Previous conversation context:\n"
        "Interaction 1:\n"
        "  User Question: What are the sales?\n"
        "  SQL Generated: SELECT sales FROM data;\n"
        "  Query Results: [(100,), (200,)]\n"
        "  Response/Analysis: Sales values are 100 and 200.\n"
        "----------------------------------------\n"
    )
    assert history_manager.get_formatted_history() == expected_format

def test_get_formatted_history_multiple_interactions(history_manager):
    """Test formatting history with multiple interactions, including missing parts."""
    history_manager.add_interaction(question="Hi", response_type="direct_answer", analysis="Hello!")
    history_manager.add_interaction(question="Need sales", response_type="clarification_needed", analysis="Which region?")
    history_manager.add_interaction(question="Sales North?", response_type="sql_analysis", sql_query="SQL North", results="[(50,)]", analysis="North: 50")

    formatted = history_manager.get_formatted_history()

    assert "Interaction 1:" in formatted
    assert "User Question: Hi" in formatted
    assert "Response/Analysis: Hello!" in formatted
    assert "SQL Generated:" not in formatted.split("Interaction 1:")[1].split("Interaction 2:")[0] # Check SQL is not in Int 1

    assert "Interaction 2:" in formatted
    assert "User Question: Need sales" in formatted
    assert "Response/Analysis: Which region?" in formatted
    assert "SQL Generated:" not in formatted.split("Interaction 2:")[1].split("Interaction 3:")[0] # Check SQL is not in Int 2

    assert "Interaction 3:" in formatted
    assert "User Question: Sales North?" in formatted
    assert "SQL Generated: SQL North" in formatted
    assert "Query Results: [(50,)]" in formatted
    assert "Response/Analysis: North: 50" in formatted
    assert formatted.count("----------------------------------------") == 3 