from pathlib import Path

# Project root = parent directory of src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DATA_DIR = PROJECT_ROOT / "data"
VECTOR_DIR = PROJECT_ROOT / "vector_store"

CHUNK_SIZE = 450
CHUNK_OVERLAP = 80
TOP_K = 4

MAX_NEW_TOKENS = 350
TEMPERATURE = 0.2
TOP_P = 0.9