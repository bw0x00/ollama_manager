import argparse
import configparser
import os
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

    def download_model(self, model_name: str):
        """
        Placeholder method to handle the model download process.
        """
        print(f"--- Initiating Download Process ---")
        print(f"Attempting to download model: {model_name}")
        print(f"Configuration loaded successfully. Example setting: {self.config.get('manifests', 'N/A')}")
        # Future download logic will go here

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
