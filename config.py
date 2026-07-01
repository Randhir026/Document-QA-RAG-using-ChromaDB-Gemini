import os
from dotenv import load_dotenv

load_dotenv()

# Gemini API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Folder containing documents
DOCUMENT_FOLDER = "documents"

# Chroma Database
CHROMA_DB_DIR = "chroma_db"

# Embedding Model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Gemini Model
GEMINI_MODEL = "gemini-2.5-flash"

# Chunk Settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100