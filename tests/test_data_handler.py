import os
import pytest
from src.app.data_handler import _get_schema_cache_path, SCHEMA_ANALYSIS_CACHE_DIR

# Ensure the cache directory exists for testing functions that might use it
# (though _get_schema_cache_path doesn't strictly need it to exist)
@pytest.fixture(scope="module", autouse=True)
def ensure_cache_dir():
    os.makedirs(SCHEMA_ANALYSIS_CACHE_DIR, exist_ok=True)
    yield
    # Optional: Cleanup cache dir after tests if needed, but generally okay to leave it
    # if os.path.exists(SCHEMA_ANALYSIS_CACHE_DIR):
    #     # Be careful with rmtree in tests!
    #     # import shutil
    #     # shutil.rmtree(SCHEMA_ANALYSIS_CACHE_DIR)
    #     pass

def test_get_schema_cache_path_simple():
    """Test with a simple db filename."""
    db_path = "my_database.db"
    expected_filename = "my_database.db.json"
    expected_path = os.path.join(SCHEMA_ANALYSIS_CACHE_DIR, expected_filename)
    assert _get_schema_cache_path(db_path) == expected_path

def test_get_schema_cache_path_with_path():
    """Test with a db path including directories."""
    db_path = "data/output/another_db.sqlite"
    expected_filename = "another_db.sqlite.json"
    expected_path = os.path.join(SCHEMA_ANALYSIS_CACHE_DIR, expected_filename)
    assert _get_schema_cache_path(db_path) == expected_path

def test_get_schema_cache_path_absolute_path():
    """Test with an absolute db path."""
    db_path = "/users/test/data/final.db"
    expected_filename = "final.db.json"
    expected_path = os.path.join(SCHEMA_ANALYSIS_CACHE_DIR, expected_filename)
    assert _get_schema_cache_path(db_path) == expected_path

def test_get_schema_cache_path_no_extension():
    """Test with a db filename that has no extension (unlikely but possible)."""
    db_path = "database_no_ext"
    expected_filename = "database_no_ext.json"
    expected_path = os.path.join(SCHEMA_ANALYSIS_CACHE_DIR, expected_filename)
    assert _get_schema_cache_path(db_path) == expected_path 