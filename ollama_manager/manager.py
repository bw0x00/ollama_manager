import argparse
import configparser
import os
import urllib.request
import urllib.error
import json
import hashlib
import time
import socket
import shutil
import sys
from typing import Dict

class ModelManager:
    """
    Manages the process of downloading and interacting with Ollama models.
    Reads configuration from config/ollama.conf.
    """
    def __init__(self, config_path: str = "config/ollama.conf", output_dir: str = "output"):
        self.config_path = config_path
        self.output_dir = output_dir
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
                target_dir = os.path.join(self.output_dir, cleaned_path)
                
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
        save_dir = os.path.join(self.output_dir, manifest_base, name)
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

    def download_blob(self, model_name: str, digest: str, expected_size: int, url_template_key: str) -> bool:
        """
        Downloads a blob (config or layer), verifies its size and SHA256 digest,
        and saves it to the blobs directory.
        """
        if ':' in model_name:
            name, _ = model_name.split(':', 1)
        else:
            name = model_name

        url_template = self.config.get(url_template_key)
        if not url_template:
            print(f"Error: '{url_template_key}' URL template not found.")
            return False

        # Replace placeholders. We replace both $config and $layer with the digest.
        url = url_template.replace('$name', name).replace('$config', digest).replace('$layer', digest)
        
        # Determine save path
        blobs_base = self.config.get('blobs', '').replace('~/.ollama/', '').lstrip('/')
        save_dir = os.path.join(self.output_dir, blobs_base)
        os.makedirs(save_dir, exist_ok=True)
        
        free_space = shutil.disk_usage(save_dir).free
        if free_space < expected_size:
            print(f"   ❌ Error: Not enough free disk space. Required: {expected_size / (1024*1024):.2f} MB, Free: {free_space / (1024*1024):.2f} MB")
            sys.exit(1)

        # Format filename: replace ':' with '-' (e.g., sha256:123 -> sha256-123)
        filename = digest.replace(':', '-')
        save_path = os.path.join(save_dir, filename)

        print(f"Downloading blob {digest} ({expected_size / (1024*1024):.2f} MB)...")
        
        downloaded_size = 0
        if os.path.exists(save_path):
            downloaded_size = os.path.getsize(save_path)
            if downloaded_size > expected_size:
                os.remove(save_path)
                downloaded_size = 0

        last_printed_mb = downloaded_size // (1024 * 1024)
        
        while downloaded_size < expected_size:
            try:
                req = urllib.request.Request(url)
                if downloaded_size > 0:
                    req.add_header('Range', f'bytes={downloaded_size}-')
                    
                # Use a timeout to detect stalled downloads
                with urllib.request.urlopen(req, timeout=15) as response, open(save_path, 'ab') as out_file:
                    while True:
                        chunk = response.read(8192 * 4)
                        if not chunk:
                            break
                        out_file.write(chunk)
                        downloaded_size += len(chunk)
                        
                        current_mb = downloaded_size // (1024 * 1024)
                        if current_mb - last_printed_mb >= 50:
                            print(f"   ... downloaded {current_mb} MB")
                            last_printed_mb = current_mb
                            
            except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError) as e:
                print(f"   ⚠️ Download stalled or error ({e}). Retrying in 10 seconds...")
                time.sleep(10)
                continue
            except Exception as e:
                print(f"   ❌ Unexpected error: {e}")
                return False

        if downloaded_size != expected_size:
            print(f"   ❌ Size mismatch for {digest}: expected {expected_size}, got {downloaded_size}")
            return False
            
        print(f"   Verifying digest for {digest}...")
        sha256_hash = hashlib.sha256()
        with open(save_path, 'rb') as f:
            while True:
                chunk = f.read(8192 * 4)
                if not chunk:
                    break
                sha256_hash.update(chunk)
                
        actual_digest = f"sha256:{sha256_hash.hexdigest()}"
        
        if actual_digest != digest:
            print(f"   ❌ Digest mismatch for {digest}: expected {digest}, got {actual_digest}")
            os.remove(save_path)
            return False
            
        print(f"   ✅ Blob saved and verified: {save_path}")
        return True

    def download_model_files(self, model_name: str, manifest: dict):
        """
        Extracts digests from the manifest and downloads the corresponding config and layer blobs.
        """
        # Download config
        config_info = manifest.get('config', {})
        config_digest = config_info.get('digest')
        config_size = config_info.get('size')
        
        if config_digest and config_size is not None:
            self.download_blob(model_name, config_digest, config_size, 'ollama_config')
            
        # Download layers
        layers = manifest.get('layers', [])
        for layer in layers:
            layer_digest = layer.get('digest')
            layer_size = layer.get('size')
            if layer_digest and layer_size is not None:
                self.download_blob(model_name, layer_digest, layer_size, 'ollama_layer')

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

    try:
        if args.download:
            manager.download_model(args.download)
        else:
            parser.print_help()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting gracefully...")
        sys.exit(0)

if __name__ == "__main__":
    main()
