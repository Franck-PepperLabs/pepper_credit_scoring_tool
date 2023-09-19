import os
import pytest
from unittest.mock import patch
from pepper import env

def test_get_project_dir_normal():
    # Normal operation test
    returned_path = env.get_project_dir()
    test_path = os.path.dirname(__file__)
    expected_path = os.path.abspath(os.path.join(test_path, "../../.."))
    _, returned_path = os.path.splitdrive(returned_path)
    _, expected_path = os.path.splitdrive(expected_path)
    assert returned_path == expected_path

@patch.dict(os.environ, {}, clear=True)
def test_get_project_dir_env_not_set():
    # Test when the environment variable is not set
    with pytest.raises(RuntimeError):
        env.get_project_dir()

def test_get_project_dir_invalid_path():
    # Test when the environment variable points to a nonexistent folder
    os.environ["PROJECT_DIR"] = "/path/to/nonexistent/folder"
    with pytest.raises(FileNotFoundError):
        env.get_project_dir()
