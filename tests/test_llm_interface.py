import pytest
from unittest.mock import patch, MagicMock

# Import the function to test
from src.app.llm_interface import generate_chat_title, _call_llm

# Mock LLM object (can be simple if only type matters for the tested function)
# If the function inspected the llm object, we'd need a more sophisticated mock.
MockLlm = MagicMock()


# Use patch to replace _call_llm within the llm_interface module for the duration of the test
@patch("src.app.llm_interface._call_llm")
def test_generate_chat_title_success(mock_call_llm):
    """Test successful title generation."""
    # Configure the mock to return a specific title when called
    mock_call_llm.return_value = "Concise Title: Test Title Example"

    question = "What are the sales figures for the western region last quarter?"
    expected_title = "Test Title Example"  # Expected title after stripping prefix

    # Call the function with the mock LLM object
    actual_title = generate_chat_title(question, MockLlm)

    # Assertions
    assert actual_title == expected_title
    # Check that _call_llm was called once
    mock_call_llm.assert_called_once()
    # Optionally check the arguments _call_llm was called with
    args, kwargs = mock_call_llm.call_args
    assert args[0] == MockLlm # Check llm object passed
    assert "User's First Question:" in args[1] # Check prompt structure
    assert question in args[1]


@patch("src.app.llm_interface._call_llm")
def test_generate_chat_title_llm_error(mock_call_llm):
    """Test title generation when the LLM call returns an error string."""
    # Configure the mock to simulate an LLM error response
    mock_call_llm.return_value = "(Error during LLM call: API Error)"

    question = "Tell me about product X."

    # Call the function
    actual_title = generate_chat_title(question, MockLlm)

    # Assertions
    assert actual_title is None # Function should return None on LLM error
    mock_call_llm.assert_called_once()


@patch("src.app.llm_interface._call_llm")
def test_generate_chat_title_empty_response(mock_call_llm):
    """Test title generation when the LLM returns an empty string."""
    # Configure the mock to return an empty string
    mock_call_llm.return_value = ""

    question = "Show me the data."

    # Call the function
    actual_title = generate_chat_title(question, MockLlm)

    # Assertions
    assert actual_title is None # Function should return None on empty response
    mock_call_llm.assert_called_once()


@patch("src.app.llm_interface._call_llm")
def test_generate_chat_title_strips_prefix_and_quotes(mock_call_llm):
    """Test that the prefix and surrounding quotes are stripped correctly."""
    # Test various return formats from the mock LLM
    test_cases = [
        ("Concise Title: Simple Case", "Simple Case"),
        ('Concise Title: "Quoted Title" ', 'Quoted Title'), # With quotes and space
        ("\'Single Quoted\'", "Single Quoted"), # No prefix, just quotes
        (" No Prefix Title ", "No Prefix Title"), # Just surrounding spaces
        ("Concise Title: ", ""), # Prefix but empty title
    ]

    for llm_return, expected in test_cases:
        mock_call_llm.reset_mock() # Reset mock for each case
        mock_call_llm.return_value = llm_return
        question = "A test question"
        actual_title = generate_chat_title(question, MockLlm)
        assert actual_title == expected, f"Failed for LLM return: '{llm_return}'"
        mock_call_llm.assert_called_once() 