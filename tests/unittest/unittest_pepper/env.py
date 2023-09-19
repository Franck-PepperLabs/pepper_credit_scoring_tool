import os
import unittest
from unittest.mock import patch

from pepper import env

class TestEnv(unittest.TestCase):
    def setUp(self):
        # Save the current value of PROJECT_DIR if it exists
        self.original_project_dir = os.getenv("PROJECT_DIR")

    def tearDown(self):
        # Restore the original value of PROJECT_DIR after each test
        if self.original_project_dir is not None:
            os.environ["PROJECT_DIR"] = self.original_project_dir
        else:
            del os.environ["PROJECT_DIR"]

    def test_get_project_dir_normal(self):
        # Normal operation test
        returned_path = env.get_project_dir()
        test_path = os.path.dirname(__file__)
        expected_path = os.path.abspath(os.path.join(test_path, "../../.."))
        _, returned_path = os.path.splitdrive(returned_path)
        _, expected_path = os.path.splitdrive(expected_path)
        self.assertEqual(returned_path, expected_path)

    @patch.dict(os.environ, {}, clear=True)
    def test_get_project_dir_env_not_set(self):
        # Test when the environment variable is not set
        with self.assertRaises(RuntimeError):
            env.get_project_dir()

    def test_get_project_dir_invalid_path(self):
        # Test when the environment variable points to a nonexistent folder
        os.environ["PROJECT_DIR"] = "/path/to/nonexistent/folder"
        with self.assertRaises(FileNotFoundError):
            env.get_project_dir()

if __name__ == "__main__":
    unittest.main()
