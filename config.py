import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
VECTOR_DB_PATH = PROJECT_ROOT / "vector_db"

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Model settings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
GEMINI_MODEL = "gemini-2.5-flash"

# Chunking settings
CHUNK_SIZE = 200
CHUNK_OVERLAP = 20

# Retrieval settings
TOP_K_RESULTS = 5

# Create directories if they don't exist
DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)
