import os
from dotenv import load_dotenv
 
load_dotenv()
 
PAYMENT_HOST = os.getenv("PAYMENT_HOST", "127.0.0.1")
PAYMENT_PORT = os.getenv("PAYMENT_PORT", "50051")
 
# Agent HTTP server
AGENT_HOST = os.getenv("AGENT_HOST", "0.0.0.0")
AGENT_PORT = int(os.getenv("AGENT_PORT", "8001"))
 
LLAMA_MODEL = os.getenv("LLAMA_MODEL", "llama3.1")