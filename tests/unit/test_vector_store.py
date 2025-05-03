import pytest
import os
from unittest.mock import patch, MagicMock, call

# Class to test
from src.app.vector_store import VectorStoreManager

# Import classes for spec mocking
from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.base.base_query_engine import BaseQueryEngine


@pytest.fixture
def mock_dependencies(mocker):
    """Fixture to mock external dependencies."""
    # Mock LlamaIndex classes/functions
    mock_vector_index_cls = mocker.patch("src.app.vector_store.VectorStoreIndex")
    mock_storage_ctx_cls = mocker.patch("src.app.vector_store.StorageContext")
    mock_load_index_func = mocker.patch("src.app.vector_store.load_index_from_storage")

    # Mock filesystem functions
    mock_exists = mocker.patch("src.app.vector_store.os.path.exists")
    mock_makedirs = mocker.patch("src.app.vector_store.os.makedirs")

    # Mock base embedding class (used for type hints and potentially checks)
    mock_embed_model = MagicMock(spec=BaseEmbedding)

    # Mock documents list
    mock_docs = [MagicMock(spec=Document), MagicMock(spec=Document)]

    return {
        "VectorStoreIndex": mock_vector_index_cls,
        "StorageContext": mock_storage_ctx_cls,
        "load_index_from_storage": mock_load_index_func,
        "exists": mock_exists,
        "makedirs": mock_makedirs,
        "embed_model": mock_embed_model,
        "docs": mock_docs
    }

@pytest.fixture
def store_manager(tmp_path): # Use tmp_path for persist_dir
    """Provides a VectorStoreManager instance with a temporary persist directory."""
    persist_dir = tmp_path / "vector_cache"
    return VectorStoreManager(persist_dir=str(persist_dir))

# --- Tests for _is_cache_valid --- #

def test_is_cache_valid_true(store_manager, mock_dependencies):
    """Test cache is valid when directory and key file exist."""
    mock_dependencies["exists"].return_value = True # Both dir and file exist
    assert store_manager._is_cache_valid() is True
    # Check calls to os.path.exists
    expected_calls = [
        call(store_manager.persist_dir),
        call(os.path.join(store_manager.persist_dir, 'docstore.json'))
    ]
    mock_dependencies["exists"].assert_has_calls(expected_calls, any_order=False)

def test_is_cache_valid_dir_missing(store_manager, mock_dependencies):
    """Test cache is invalid when directory is missing."""
    mock_dependencies["exists"].return_value = False # Dir missing
    assert store_manager._is_cache_valid() is False
    mock_dependencies["exists"].assert_called_once_with(store_manager.persist_dir)

def test_is_cache_valid_file_missing(store_manager, mock_dependencies):
    """Test cache is invalid when key file (docstore.json) is missing."""
    # Simulate dir exists, but file doesn't
    mock_dependencies["exists"].side_effect = lambda p: p == store_manager.persist_dir
    assert store_manager._is_cache_valid() is False
    # Check calls
    expected_calls = [
        call(store_manager.persist_dir),
        call(os.path.join(store_manager.persist_dir, 'docstore.json'))
    ]
    mock_dependencies["exists"].assert_has_calls(expected_calls, any_order=False)


# --- Tests for _build_and_persist_index --- #

def test_build_and_persist_success(store_manager, mock_dependencies):
    """Test successful index building and persistence."""
    # Mock the index object returned by from_documents
    mock_index_instance = MagicMock(spec=VectorStoreIndex)
    mock_storage_context_instance = MagicMock()
    mock_index_instance.storage_context = mock_storage_context_instance
    mock_dependencies["VectorStoreIndex"].from_documents.return_value = mock_index_instance

    index = store_manager._build_and_persist_index(
        schema_docs=mock_dependencies["docs"],
        embed_model=mock_dependencies["embed_model"]
    )

    assert index is mock_index_instance
    mock_dependencies["makedirs"].assert_called_once_with(store_manager.persist_dir, exist_ok=True)
    mock_dependencies["VectorStoreIndex"].from_documents.assert_called_once_with(
        mock_dependencies["docs"],
        embed_model=mock_dependencies["embed_model"]
    )
    mock_storage_context_instance.persist.assert_called_once_with(persist_dir=store_manager.persist_dir)

def test_build_and_persist_no_embed_model(store_manager):
    """Test build fails if no embed model is provided."""
    index = store_manager._build_and_persist_index([], None)
    assert index is None

def test_build_and_persist_no_docs(store_manager, mock_dependencies):
    """Test build fails if no documents are provided."""
    index = store_manager._build_and_persist_index([], mock_dependencies["embed_model"])
    assert index is None

def test_build_and_persist_build_exception(store_manager, mock_dependencies):
    """Test build handles exception during VectorStoreIndex.from_documents."""
    mock_dependencies["VectorStoreIndex"].from_documents.side_effect = Exception("Build Error")
    index = store_manager._build_and_persist_index(mock_dependencies["docs"], mock_dependencies["embed_model"])
    assert index is None
    mock_dependencies["VectorStoreIndex"].from_documents.assert_called_once()


# --- Tests for load_or_build_index --- #

def test_load_or_build_cache_hit(store_manager, mock_dependencies):
    """Test loading from a valid cache."""
    # Simulate cache is valid
    store_manager._is_cache_valid = MagicMock(return_value=True)
    # Mock loaded index and query engine
    mock_index_instance = MagicMock(spec=VectorStoreIndex)
    mock_query_engine_instance = MagicMock(spec=BaseQueryEngine)
    mock_index_instance.as_query_engine.return_value = mock_query_engine_instance
    mock_dependencies["load_index_from_storage"].return_value = mock_index_instance
    mock_storage_context_instance = MagicMock()
    mock_dependencies["StorageContext"].from_defaults.return_value = mock_storage_context_instance

    index, query_engine = store_manager.load_or_build_index(
        schema_docs=mock_dependencies["docs"],
        embed_model=mock_dependencies["embed_model"],
        force_rebuild=False
    )

    assert index is mock_index_instance
    assert query_engine is mock_query_engine_instance
    store_manager._is_cache_valid.assert_called_once()
    mock_dependencies["StorageContext"].from_defaults.assert_called_once_with(persist_dir=store_manager.persist_dir)
    mock_dependencies["load_index_from_storage"].assert_called_once_with(mock_storage_context_instance, embed_model=mock_dependencies["embed_model"])
    mock_index_instance.as_query_engine.assert_called_once_with(response_mode="no_text")
    # Ensure build was NOT called
    mock_dependencies["VectorStoreIndex"].from_documents.assert_not_called()

def test_load_or_build_cache_miss_build_success(store_manager, mock_dependencies):
    """Test building index when cache is invalid."""
    store_manager._is_cache_valid = MagicMock(return_value=False) # Cache invalid
    # Mock build process succeeding
    mock_index_instance = MagicMock(spec=VectorStoreIndex)
    mock_query_engine_instance = MagicMock(spec=BaseQueryEngine)
    mock_index_instance.as_query_engine.return_value = mock_query_engine_instance
    mock_storage_context_instance = MagicMock()
    mock_index_instance.storage_context = mock_storage_context_instance
    mock_dependencies["VectorStoreIndex"].from_documents.return_value = mock_index_instance

    index, query_engine = store_manager.load_or_build_index(
        schema_docs=mock_dependencies["docs"],
        embed_model=mock_dependencies["embed_model"],
        force_rebuild=False
    )

    assert index is mock_index_instance
    assert query_engine is mock_query_engine_instance
    store_manager._is_cache_valid.assert_called_once()
    mock_dependencies["load_index_from_storage"].assert_not_called()
    mock_dependencies["VectorStoreIndex"].from_documents.assert_called_once() # Build was called
    mock_index_instance.as_query_engine.assert_called_once_with(response_mode="no_text")

def test_load_or_build_force_rebuild(store_manager, mock_dependencies):
    """Test force_rebuild=True triggers build even if cache is valid."""
    store_manager._is_cache_valid = MagicMock(return_value=True) # Cache IS valid
    # Mock build process succeeding
    mock_index_instance = MagicMock(spec=VectorStoreIndex)
    mock_query_engine_instance = MagicMock(spec=BaseQueryEngine)
    mock_index_instance.as_query_engine.return_value = mock_query_engine_instance
    mock_storage_context_instance = MagicMock()
    mock_index_instance.storage_context = mock_storage_context_instance
    mock_dependencies["VectorStoreIndex"].from_documents.return_value = mock_index_instance

    index, query_engine = store_manager.load_or_build_index(
        schema_docs=mock_dependencies["docs"],
        embed_model=mock_dependencies["embed_model"],
        force_rebuild=True # Force rebuild
    )

    assert index is mock_index_instance
    assert query_engine is mock_query_engine_instance
    store_manager._is_cache_valid.assert_not_called() # Cache check skipped
    mock_dependencies["load_index_from_storage"].assert_not_called()
    mock_dependencies["VectorStoreIndex"].from_documents.assert_called_once() # Build was called
    mock_index_instance.as_query_engine.assert_called_once_with(response_mode="no_text")

def test_load_or_build_load_exception_then_build(store_manager, mock_dependencies):
    """Test build is attempted if loading from valid cache fails."""
    store_manager._is_cache_valid = MagicMock(return_value=True) # Cache valid
    # Simulate exception during load_index_from_storage
    mock_dependencies["load_index_from_storage"].side_effect = Exception("Load Error")
    # Mock build process succeeding
    mock_index_instance = MagicMock(spec=VectorStoreIndex)
    mock_query_engine_instance = MagicMock(spec=BaseQueryEngine)
    mock_index_instance.as_query_engine.return_value = mock_query_engine_instance
    mock_storage_context_instance = MagicMock()
    mock_index_instance.storage_context = mock_storage_context_instance
    mock_dependencies["VectorStoreIndex"].from_documents.return_value = mock_index_instance

    index, query_engine = store_manager.load_or_build_index(
        schema_docs=mock_dependencies["docs"],
        embed_model=mock_dependencies["embed_model"],
        force_rebuild=False
    )

    assert index is mock_index_instance # Should get index from successful build
    assert query_engine is mock_query_engine_instance
    store_manager._is_cache_valid.assert_called_once()
    mock_dependencies["load_index_from_storage"].assert_called_once() # Load was attempted
    mock_dependencies["VectorStoreIndex"].from_documents.assert_called_once() # Build was attempted
    mock_index_instance.as_query_engine.assert_called_once_with(response_mode="no_text")

def test_load_or_build_build_fails(store_manager, mock_dependencies):
    """Test returns None, None if building fails."""
    store_manager._is_cache_valid = MagicMock(return_value=False) # Cache miss
    # Simulate build failure
    mock_dependencies["VectorStoreIndex"].from_documents.return_value = None

    index, query_engine = store_manager.load_or_build_index(
        schema_docs=mock_dependencies["docs"],
        embed_model=mock_dependencies["embed_model"],
        force_rebuild=False
    )

    assert index is None
    assert query_engine is None
    mock_dependencies["VectorStoreIndex"].from_documents.assert_called_once() 