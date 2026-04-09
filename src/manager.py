import argparse
import configparser
import os
import urllib.request
import urllib.error
import json
import hashlib
from typing import Dict

class ModelManager:
    """
    Manages the process of downloading and interacting with Ollama models.
    Reads configuration from config/ollama.conf.
    """
    def __init__(self, config_path: str = "config/ollama.conf"):
        self.config_path = config_path
        self.config = self._load_config()
        self._ensure_output_directories()

    def _load_config(self) -> Dict[str, str]:
        """
        Loads configuration settings from the specified INI file.
        Assumes the config file is simple key=value pairs for this initial structure.
        """
        config_data = {}
        if not os.path.exists(self.config_path):
            print(f"Warning: Configuration file not found at {self.config_path}")
            return config_data

        # Simple parsing for the provided key=value format in the config file
        try:
            with open(self.config_path, 'r') as f:
                content = f.read()
            
            # Split by lines, filter out comments/empty lines, and parse key=value
            for line in content.splitlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        config_data[key.strip()] = value.strip()
        except Exception as e:
            print(f"Error reading configuration file: {e}")
            return {}
            
        return config_data

    def _ensure_output_directories(self):
        """
        Checks and creates necessary directory structures within the 'output/' folder
        based on paths defined in the configuration (manifests and blobs).
        """
        paths_to_check = ["manifests", "blobs"]
        
        for key in paths_to_check:
            if key in self.config:
                raw_path = self.config[key]
                
                # Strip the leading "~/.ollama/" segment
                if raw_path.startswith("~/.ollama/"):
                    cleaned_path = raw_path[len("~/.ollama/"):]
                else:
                    cleaned_path = raw_path
                
                # Construct the final path relative to the 'output/' directory
                # Using os.path.join to ensure cross-platform compatibility, 
                # but we need to handle the leading slash if cleaned_path has one.
                cleaned_path = cleaned_path.lstrip('/')
                target_dir = os.path.join("output", cleaned_path)
                
                # Create the directory if it doesn't exist
                try:
                    os.makedirs(target_dir, exist_ok=True)
                except OSError as e:
                    print(f"Error creating directory {target_dir}: {e}")

    def download_manifest(self, model_name: str) -> dict:
        """
        Downloads the manifest file for a given model name.
        Parses the name and tag, constructs the URL from the config, and saves the file.
        """
        # Parse model name and tag (default to 'latest' if no tag is provided)
        if ':' in model_name:
            name, tag = model_name.split(':', 1)
        else:
            name = model_name
            tag = "latest"

        # Construct the URL
        url_template = self.config.get('ollama_manifests')
        if not url_template:
            print("Error: 'ollama_manifests' URL template not found in configuration.")
            return {}

        url = url_template.replace('$name', name).replace('$tag', tag)
        print(f"Downloading manifest from: {url}")

        # Determine the save path based on the config
        manifest_base = self.config.get('manifests', '').replace('~/.ollama/', '').lstrip('/')
        save_dir = os.path.join("output", manifest_base, name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, tag)

        # Perform the download
        try:
            # Ollama registry requires specific Accept headers for manifests
            headers = {'Accept': 'application/vnd.docker.distribution.manifest.v2+json'}
            req = urllib.request.Request(url, headers=headers)
            
            with urllib.request.urlopen(req) as response:
                data = response.read()
                
            with open(save_path, 'wb') as f:
                f.write(data)
                
            print(f"   ✅ Manifest saved to {save_path}")
            return json.loads(data.decode('utf-8'))
            
        except urllib.error.URLError as e:
            print(f"   ❌ Failed to download manifest: {e}")
            return {}
        except json.JSONDecodeError:
            print(f"   ❌ Failed to parse manifest JSON.")
            return {}

    def download_model(self, model_name: str):
        """
        Handles the model download process.
        """
        print(f"--- Initiating Download Process ---")
        print(f"Attempting to download model: {model_name}")
        
        manifest = self.download_manifest(model_name)
        if not manifest:
            print("Aborting download process due to missing manifest.")
            return
            
        print("Manifest downloaded successfully. Downloading blobs...")
        self.download_model_files(model_name, manifest)
        print("Download process completed.")

def main():
    """
    Main entry point for the Model Manager CLI.
    """
    parser = argparse.ArgumentParser(
        description="A tool to download and manage AI models for Ollama.",
        epilog="Use --help for more information."
    )
    parser.add_argument(
        "--download", 
        type=str, 
        help="The name of the model to download (e.g., llama3:8b)."
    )
    
    args = parser.parse_args()

    manager = ModelManager()

    if args.download:
        manager.download_model(args.download)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
