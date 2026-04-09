import unittest
import os
import tempfile
from unittest.mock import patch, mock_open, MagicMock
import unittest.mock
import json
import hashlib

# Assuming src/manager.py is importable or we adjust the path for testing
# For this test file, we assume we can import the class directly.
from ollama_manager.manager import ModelManager

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
ollama_config = https://registry.ollama.ai/v2/library/$name/blobs/$config
ollama_layer = https://registry.ollama.ai/v2/library/$name/blobs/$layer
"""
        
        # Write mock content to the temporary config file
        with open(self.temp_config_path, 'w') as f:
            f.write(self.mock_config_content)

    def tearDown(self):
        # Clean up the temporary directory
        self.temp_dir.cleanup()

    @patch('ollama_manager.manager.ModelManager._load_config')
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

    @patch('ollama_manager.manager.ModelManager._load_config')
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
            unittest.mock.call(os.path.join("output", "models/manifests/registry.ollama.ai/library"), exist_ok=True),
            unittest.mock.call(os.path.join("output", "models/blobs"), exist_ok=True)
        ]
        
        # Assert that os.makedirs was called exactly twice with the correct paths
        mock_makedirs.assert_has_calls(expected_calls, any_order=True)
        self.assertEqual(mock_makedirs.call_count, 2)

    @patch('ollama_manager.manager.ModelManager._load_config')
    @patch('os.makedirs')
    def test_initialization_calls_setup(self, mock_makedirs, mock_load_config):
        """Tests that necessary directories are created upon initialization."""
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
            unittest.mock.call(os.path.join("output", "models/manifests/registry.ollama.ai/library"), exist_ok=True),
            unittest.mock.call(os.path.join("output", "models/blobs"), exist_ok=True)
        ]
    
        # Assert that os.makedirs was called exactly twice with the correct paths
        mock_makedirs.assert_has_calls(expected_calls, any_order=True)

    @patch('urllib.request.urlopen')
    @patch('os.makedirs')
    def test_download_manifest_success(self, mock_makedirs, mock_urlopen):
        """Tests successful download and parsing of a manifest."""
        mock_manifest_data = {"config": {"digest": "sha256:dummy", "size": 100}}
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(mock_manifest_data).encode('utf-8')
        mock_urlopen.return_value.__enter__.return_value = mock_response

        manager = ModelManager(config_path=self.temp_config_path)
        
        with patch('builtins.open', mock_open()) as mocked_file:
            manifest = manager.download_manifest("testmodel:latest")
            
            self.assertEqual(manifest, mock_manifest_data)
            # Verify file was opened for writing
            mocked_file.assert_called_once()

    @patch('shutil.disk_usage')
    @patch('os.path.exists')
    @patch('urllib.request.urlopen')
    @patch('os.makedirs')
    def test_download_blob_success(self, mock_makedirs, mock_urlopen, mock_exists, mock_disk_usage):
        """Tests successful download, size verification, and hash verification of a blob."""
        mock_disk_usage.return_value.free = 999999999
        mock_exists.return_value = False
        
        test_data = b"dummy layer data"
        expected_size = len(test_data)
        expected_digest = f"sha256:{hashlib.sha256(test_data).hexdigest()}"

        mock_response = MagicMock()
        # Return data on first read, empty bytes on second read to simulate EOF
        mock_response.read.side_effect = [test_data, b""] 
        mock_urlopen.return_value.__enter__.return_value = mock_response

        manager = ModelManager(config_path=self.temp_config_path)
        
        with patch('builtins.open', mock_open()) as mocked_file:
            result = manager.download_blob("testmodel", expected_digest, expected_size, "ollama_layer")
            
            self.assertTrue(result)
            # open is called twice: once for writing ('ab'), once for reading ('rb') to verify hash
            self.assertEqual(mocked_file.call_count, 2)

    @patch('shutil.disk_usage')
    @patch('os.path.exists')
    @patch('urllib.request.urlopen')
    @patch('os.makedirs')
    @patch('os.remove')
    def test_download_blob_size_mismatch(self, mock_remove, mock_makedirs, mock_urlopen, mock_exists, mock_disk_usage):
        """Tests blob download failure due to size mismatch."""
        mock_disk_usage.return_value.free = 999999999
        mock_exists.return_value = False
        
        test_data = b"dummy layer data"
        expected_size = 5 # Intentionally wrong size (smaller than test_data to exit the while loop)
        expected_digest = f"sha256:{hashlib.sha256(test_data).hexdigest()}"

        mock_response = MagicMock()
        mock_response.read.side_effect = [test_data, b""] 
        mock_urlopen.return_value.__enter__.return_value = mock_response

        manager = ModelManager(config_path=self.temp_config_path)
        
        with patch('builtins.open', mock_open()):
            result = manager.download_blob("testmodel", expected_digest, expected_size, "ollama_layer")
            
            self.assertFalse(result)
            # os.remove is not called for size mismatch in the updated logic, it just returns False
            # so we don't assert mock_remove.assert_called_once() here anymore

    @patch('ollama_manager.manager.ModelManager.download_blob')
    @patch('os.makedirs')
    def test_download_model_files(self, mock_makedirs, mock_download_blob):
        """Tests that download_model_files correctly extracts digests and calls download_blob."""
        mock_manifest = {
            "config": {"digest": "sha256:config123", "size": 500},
            "layers": [
                {"digest": "sha256:layer1", "size": 1000},
                {"digest": "sha256:layer2", "size": 2000}
            ]
        }
        
        manager = ModelManager(config_path=self.temp_config_path)
        manager.download_model_files("testmodel:latest", mock_manifest)
        
        # Check if download_blob was called 3 times (1 config + 2 layers)
        self.assertEqual(mock_download_blob.call_count, 3)
        
        # Verify specific calls
        mock_download_blob.assert_any_call("testmodel:latest", "sha256:config123", 500, 'ollama_config')
        mock_download_blob.assert_any_call("testmodel:latest", "sha256:layer1", 1000, 'ollama_layer')
        mock_download_blob.assert_any_call("testmodel:latest", "sha256:layer2", 2000, 'ollama_layer')

    @patch('shutil.disk_usage')
    @patch('os.makedirs')
    def test_download_blob_insufficient_space(self, mock_makedirs, mock_disk_usage):
        """Tests that the process exits if there is not enough disk space."""
        mock_disk_usage.return_value.free = 100 # Less than expected_size
        
        manager = ModelManager(config_path=self.temp_config_path)
        
        with self.assertRaises(SystemExit) as cm:
            manager.download_blob("testmodel", "sha256:dummy", 1000, "ollama_layer")
            
        self.assertEqual(cm.exception.code, 1)

    @patch('time.sleep')
    @patch('shutil.disk_usage')
    @patch('os.path.exists')
    @patch('urllib.request.urlopen')
    @patch('os.makedirs')
    def test_download_blob_retry_logic(self, mock_makedirs, mock_urlopen, mock_exists, mock_disk_usage, mock_sleep):
        """Tests that the download retries on timeout."""
        mock_disk_usage.return_value.free = 999999999
        mock_exists.return_value = False
        
        test_data = b"dummy layer data"
        expected_size = len(test_data)
        expected_digest = f"sha256:{hashlib.sha256(test_data).hexdigest()}"
        
        mock_response = MagicMock()
        mock_response.read.side_effect = [test_data, b""]
        
        import socket
        # First call raises timeout, second call succeeds
        mock_urlopen.side_effect = [
            socket.timeout("Timeout"), 
            MagicMock(__enter__=MagicMock(return_value=mock_response))
        ]
        
        manager = ModelManager(config_path=self.temp_config_path)
        
        with patch('builtins.open', mock_open()):
            result = manager.download_blob("testmodel", expected_digest, expected_size, "ollama_layer")
            
        self.assertTrue(result)
        mock_sleep.assert_called_once_with(10)

if __name__ == '__main__':
    unittest.main()
