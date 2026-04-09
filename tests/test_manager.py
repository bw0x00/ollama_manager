import unittest
import os
import tempfile
from unittest.mock import patch, mock_open, MagicMock

# Assuming src/manager.py is importable or we adjust the path for testing
# For this test file, we assume we can import the class directly.
from src.manager import ModelManager

class TestModelManager(unittest.TestCase):

    def setUp(self):
        # Setup a temporary directory structure for testing file operations
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_config_path = os.path.join(self.temp_dir.name, "ollama.conf")
        
        # Mock the existence of the config file for testing purposes
        self.mock_config_content = """
# Ollama Configuration
manifests = ~/.ollama/models/manifests/registry.ollama.ai/library
blobs = ~/.ollama/models/blobs
ollama_manifests = https://registry.ollama.ai/v2/library/$name/manifests/$tag
"""
        
        # Write mock content to the temporary config file
        with open(self.temp_config_path, 'w') as f:
            f.write(self.mock_config_content)

    def tearDown(self):
        # Clean up the temporary directory
        self.temp_dir.cleanup()

    @patch('src.manager.ModelManager._load_config')
    def test_load_config_success(self, mock_load_config):
        """Tests successful loading of configuration data."""
        expected_config = {
            'manifests': '~/.ollama/models/manifests/registry.ollama.ai/library',
            'blobs': '~/.ollama/models/blobs',
            'ollama_manifests': 'https://registry.ollama.ai/v2/library/$name/manifests/$tag',
            'ollama_config': 'https://registry.ollama.ai/v2/library/$name/blobs/$config',
            'ollama_layer': 'https://registry.ollama.ai/v2/library/$name/blobs/$layer',
        }
        mock_load_config.return_value = expected_config
        
        manager = ModelManager(config_path=self.temp_config_path)
        self.assertEqual(manager.config, expected_config)

    @patch('src.manager.ModelManager._load_config')
    @patch('os.makedirs')
    def test_ensure_output_directories_creation(self, mock_makedirs, mock_load_config):
        """Tests that necessary directories are created in the correct structure."""
        # Mock the config to ensure the paths are available
        mock_load_config.return_value = {
            'manifests': '~/.ollama/models/manifests/registry.ollama.ai/library',
            'blobs': '~/.ollama/models/blobs',
            'other_key': 'should_be_ignored'
        }
        
        # Initialize the manager, which triggers _ensure_output_directories
        manager = ModelManager(config_path=self.temp_config_path)
        
        # Expected paths after stripping "~/.ollama/" and prepending "output/"
        expected_calls = [
            unittest.mock.call(os.path.join("output", "models/manifests/registry.ollama.ai/library")),
            unittest.mock.call(os.path.join("output", "models/blobs"))
        ]
        
        # Assert that os.makedirs was called exactly twice with the correct paths
        mock_makedirs.assert_has_calls(expected_calls, any_order=True)
        self.assertEqual(mock_makedirs.call_count, 2)

    @patch('src.manager.ModelManager._load_config')
    @patch('src.manager.ModelManager._ensure_output_directories')
    def test_initialization_calls_setup(self, mock_ensure_dirs, mock_load_config):
        """Tests that __init__ calls both config loading and directory setup."""
        mock_load_config.return_value = {}
        
        ModelManager(config_path=self.temp_config_path)
        
        mock_load_config.assert_called_once()
        mock_ensure_dirs.assert_called_once()

if __name__ == '__main__':
    unittest.main()
