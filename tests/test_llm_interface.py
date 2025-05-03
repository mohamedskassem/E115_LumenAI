import pytest
from unittest.mock import patch, MagicMock

# Import the function to test
from src.app.llm_interface import (
    generate_chat_title, _call_llm, validate_question, generate_sql, generate_analysis,
    LlmType, OpenAI, genai, VALIDATION_PROMPT_TEMPLATE_STR, SQL_GENERATION_PROMPT_TEMPLATE_STR,
    ANALYSIS_PROMPT_TEMPLATE_STR
)

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


# --- Tests for _call_llm (Helper Function) ---

def test_call_llm_openai():
    """Test calling _call_llm with a mocked OpenAI object."""
    mock_llm = MagicMock(spec=OpenAI)
    prompt = "Test prompt"
    expected_response = " OpenAI response "
    mock_llm.predict.return_value = expected_response

    response = _call_llm(mock_llm, prompt)

    assert response == expected_response.strip()
    mock_llm.predict.assert_called_once_with(prompt)

def test_call_llm_gemini_success():
    """Test calling _call_llm with a mocked Gemini object (success)."""
    mock_llm = MagicMock(spec=genai.GenerativeModel)
    prompt = "Test prompt"
    expected_response = " Gemini response "
    # Simulate a successful response object from genai
    mock_gemini_response = MagicMock()
    mock_gemini_response.parts = [True] # Indicate parts exist
    mock_gemini_response.text = expected_response
    mock_llm.generate_content.return_value = mock_gemini_response

    response = _call_llm(mock_llm, prompt)

    assert response == expected_response.strip()
    mock_llm.generate_content.assert_called_once_with(prompt)

def test_call_llm_gemini_blocked():
    """Test calling _call_llm with a mocked Gemini object (blocked response)."""
    mock_llm = MagicMock(spec=genai.GenerativeModel)
    prompt = "Test prompt"
    # Simulate a blocked/empty response object from genai
    mock_gemini_response = MagicMock()
    mock_gemini_response.parts = [] # Indicate NO parts
    mock_gemini_response.text = None
    mock_gemini_response.prompt_feedback = "Blocked due to safety."
    mock_llm.generate_content.return_value = mock_gemini_response

    response = _call_llm(mock_llm, prompt)

    assert response == "(LLM response blocked or empty)"
    mock_llm.generate_content.assert_called_once_with(prompt)

def test_call_llm_unsupported_type():
    """Test calling _call_llm with an unsupported LLM type."""
    mock_llm = MagicMock() # Generic mock without spec
    prompt = "Test prompt"

    response = _call_llm(mock_llm, prompt)

    assert "(Error during LLM call:" in response
    assert "Unsupported LLM type" in response

def test_call_llm_api_exception():
    """Test calling _call_llm when the underlying API call fails."""
    mock_llm = MagicMock(spec=OpenAI)
    prompt = "Test prompt"
    error_message = "API connection error"
    mock_llm.predict.side_effect = Exception(error_message)

    response = _call_llm(mock_llm, prompt)

    assert "(Error during LLM call:" in response
    assert error_message in response
    mock_llm.predict.assert_called_once_with(prompt)


# --- Tests for validate_question --- #

@patch("src.app.llm_interface._call_llm")
def test_validate_question_sql_needed(mock_call_llm):
    """Test validate_question returns SQL_NEEDED."""
    mock_call_llm.return_value = "SQL_NEEDED:"
    mock_llm = MagicMock()

    result = validate_question("q", "s", "a", "h", mock_llm)

    assert result == {"action": "SQL_NEEDED", "details": ""}
    mock_call_llm.assert_called_once()
    # Check prompt formatting (optional but good)
    prompt_arg = mock_call_llm.call_args[0][1]
    assert "Database Schema Summary:\ns" in prompt_arg
    assert "Detailed Schema Analysis (Generated by an LLM):\na" in prompt_arg
    assert "Conversation History:\nh" in prompt_arg
    assert "User Question: \"q\"" in prompt_arg

@patch("src.app.llm_interface._call_llm")
def test_validate_question_direct_answer(mock_call_llm):
    """Test validate_question returns DIRECT_ANSWER with details."""
    mock_call_llm.return_value = "DIRECT_ANSWER: Hello!"
    mock_llm = MagicMock()

    result = validate_question("q", "s", "a", "h", mock_llm)

    assert result == {"action": "DIRECT_ANSWER", "details": "Hello!"}
    mock_call_llm.assert_called_once()

@patch("src.app.llm_interface._call_llm")
def test_validate_question_clarification(mock_call_llm):
    """Test validate_question returns CLARIFICATION_NEEDED with details."""
    mock_call_llm.return_value = "CLARIFICATION_NEEDED: Specify date."
    mock_llm = MagicMock()

    result = validate_question("q", "s", "a", "h", mock_llm)

    assert result == {"action": "CLARIFICATION_NEEDED", "details": "Specify date."}
    mock_call_llm.assert_called_once()

@patch("src.app.llm_interface._call_llm")
def test_validate_question_unexpected_format(mock_call_llm):
    """Test validate_question handles unexpected LLM response format."""
    unexpected_response = "UNEXPECTED_ACTION_LABEL"
    mock_call_llm.return_value = unexpected_response
    mock_llm = MagicMock()

    result = validate_question("q", "s", "a", "h", mock_llm)

    # Should default to SQL_NEEDED and include the bad response in details
    assert result["action"] == "SQL_NEEDED"
    assert unexpected_response in result["details"]
    mock_call_llm.assert_called_once()

@patch("src.app.llm_interface._call_llm")
def test_validate_question_llm_call_error(mock_call_llm):
    """Test validate_question handles errors from _call_llm."""
    error_response = "(Error during LLM call: Test Error)"
    mock_call_llm.return_value = error_response
    mock_llm = MagicMock()

    result = validate_question("q", "s", "a", "h", mock_llm)

    # Should default to SQL_NEEDED and include the error in details
    assert result["action"] == "SQL_NEEDED"
    assert error_response in result["details"]
    mock_call_llm.assert_called_once()


# --- Tests for generate_sql --- #

@patch("src.app.llm_interface._call_llm")
def test_generate_sql_success(mock_call_llm):
    """Test successful SQL generation."""
    expected_sql = "SELECT * FROM test" # Note: no trailing semicolon expected from cleanup
    raw_llm_response = f"  ```sql\n{expected_sql};\n```  " # Only ONE semicolon here
    mock_call_llm.return_value = raw_llm_response
    mock_llm = MagicMock()

    sql, raw = generate_sql("q", "s", "a", "h", "c", mock_llm)

    assert sql == expected_sql
    assert raw == raw_llm_response
    mock_call_llm.assert_called_once()
    # Check prompt formatting (optional)
    prompt_arg = mock_call_llm.call_args[0][1]
    assert "Database Schema:\ns" in prompt_arg
    assert "Detailed Schema Analysis (Generated by an LLM):\na" in prompt_arg
    assert "Conversation History:\nh" in prompt_arg
    assert "Retrieved Context (Information potentially relevant to the user query):\nc" in prompt_arg
    assert "USER INPUT: q" in prompt_arg
    assert prompt_arg.endswith("GENERATED SQL:")

@patch("src.app.llm_interface._call_llm")
def test_generate_sql_empty_response(mock_call_llm):
    """Test SQL generation when LLM returns empty string."""
    mock_call_llm.return_value = ""
    mock_llm = MagicMock()

    sql, raw = generate_sql("q", "s", "a", "h", "c", mock_llm)

    assert sql is None
    assert "empty SQL query" in raw
    mock_call_llm.assert_called_once()

@patch("src.app.llm_interface._call_llm")
def test_generate_sql_llm_error(mock_call_llm):
    """Test SQL generation when _call_llm returns an error."""
    error_response = "(Error during LLM call: Test Error)"
    mock_call_llm.return_value = error_response
    mock_llm = MagicMock()

    sql, raw = generate_sql("q", "s", "a", "h", "c", mock_llm)

    assert sql is None
    assert raw == f"LLM Error: {error_response}"
    mock_call_llm.assert_called_once()


# --- Tests for generate_chat_title --- #

# Mock LLM object (can be simple if only type matters for the tested function)
# If the function inspected the llm object, we'd need a more sophisticated mock.
MockLlmChat = MagicMock() 