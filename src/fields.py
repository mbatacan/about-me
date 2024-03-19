from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
HF_API_KEY = os.getenv("HUGGING_FACE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
USE_SERVERLESS = False
EMBEDDING_SIZE = 384
SECRET_KEY = os.getenv("SECRET_KEY")
NAME = "Matthew Batacan"
