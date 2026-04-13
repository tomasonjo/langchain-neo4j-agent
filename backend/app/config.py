import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Neo4j MCP Server
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "neo4j")

# Neo4j Agent Memory
NEO4J_MEMORY_URI = os.environ.get("NEO4J_MEMORY_URI", NEO4J_URI)
NEO4J_MEMORY_USERNAME = os.environ.get("NEO4J_MEMORY_USERNAME", NEO4J_USERNAME)
NEO4J_MEMORY_PASSWORD = os.environ.get("NEO4J_MEMORY_PASSWORD", NEO4J_PASSWORD)
