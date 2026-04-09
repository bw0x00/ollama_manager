# Ollama Model Manager

## 🚀 Project Description
This tool is a comprehensive CLI downloader and manager for AI models intended for use with the Ollama framework. It streamlines the process of acquiring, verifying, and managing model assets, ensuring a consistent and reliable setup for local LLM development. It directly interacts with the Ollama V2 Docker Registry API to fetch manifests and blobs, verifying them against SHA256 checksums and expected file sizes.

## ✨ Features
- **Direct Registry Downloads:** Fetches model manifests and layer blobs directly from the Ollama registry.
- **Integrity Verification:** Automatically verifies downloaded blobs against expected SHA256 hashes and file sizes.
- **Ollama Directory Structure:** Replicates the native Ollama `~/.ollama/models` directory structure inside a local `output/` folder.
- **Zero Dependencies:** Built entirely using Python's standard library (`urllib`, `hashlib`, `json`, `argparse`). No `pip install` required!

## 📂 Project Structure
- `ollama_manager/`: Contains the core Python source code (`manager.py`).
- `tests/`: Houses all unit tests (`test_manager.py`).
- `docs/`: Contains additional documentation files.
- `config/`: Stores configuration files (`ollama.conf`).
- `example/`: Contains example model assets or configurations.
- `output/`: (Generated) The destination folder for downloaded models.

## ⚙️ Configuration
The tool relies on `config/ollama.conf` to determine download URLs and local storage paths. 

Example `config/ollama.conf`:
```ini
manifests = ~/.ollama/models/manifests/registry.ollama.ai/library
blobs = ~/.ollama/models/blobs
ollama_manifests = https://registry.ollama.ai/v2/library/$name/manifests/$tag
ollama_config = https://registry.ollama.ai/v2/library/$name/blobs/$config
ollama_layer = https://registry.ollama.ai/v2/library/$name/blobs/$layer
```
*Note: The tool automatically strips the `~/.ollama/` prefix and places the files inside the local `output/` directory.*

## 💻 Usage

Run the manager from the root of the repository using Python:

```bash
# View help and available arguments
python3 -m ollama_manager.manager --help

# Download a specific model (defaults to 'latest' tag if omitted)
python3 -m ollama_manager.manager --download gemma:2b
python3 -m ollama_manager.manager --download llama3
```

During the download, the tool will display progress and verify each blob:
```text
--- Initiating Download Process ---
Attempting to download model: gemma:2b
Downloading manifest from: https://registry.ollama.ai/v2/library/gemma/manifests/2b
   ✅ Manifest saved to output/models/manifests/registry.ollama.ai/library/gemma/2b
Manifest downloaded successfully. Downloading blobs...
Downloading blob sha256:c1864a5eb19305c40519da12cc543519e48a0697ecd30e15d5ac228644957d12 (1.67 MB)...
   ✅ Blob saved and verified: output/models/blobs/sha256-c1864a5eb19305c40519da12cc543519e48a0697ecd30e15d5ac228644957d12
...
Download process completed.
```

## 🧪 Testing
The project includes a comprehensive suite of unit tests that mock network and filesystem interactions.

To run the tests:
```bash
python3 -m unittest discover -s tests -v
```
