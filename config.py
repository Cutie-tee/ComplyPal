# config.py

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# File and Directory Paths
PDF_DIR = "sources/pdfs"
CHROMA_DB_PATH = "./chroma_db"
LOG_DIR = "logs"

# Model Configurations
GEMINI_CONFIG = {
    "model": "gemini-pro",
    "temperature": 0.5,
}

EMBEDDING_CONFIG = {
    "model": "models/embedding-001",
}

# Vector Store Configurations
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# RAG Configurations
RETRIEVAL_K = 3  # Number of documents to retrieve

# Evaluation Configurations
SCORE_WEIGHTS = {
    "factual": 0.4,
    "completeness": 0.4,
    "clarity": 0.2
}

# Ensure required directories exist
for directory in [PDF_DIR, CHROMA_DB_PATH, LOG_DIR]:
    os.makedirs(directory, exist_ok=True)