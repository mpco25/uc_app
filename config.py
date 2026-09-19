from pathlib import Path

INDEX_FILE = Path("data/uc_metadata.index")
METADATA_FILE = Path("data/uc_metadata.json")

OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text:latest"

REQUEST_TIMEOUT = 600

PREFERRED_MODEL = "qwen3.8:latest"

DEFAULT_CANDIDATES = 25