from langchain_openai import ChatOpenAI
from langchain import hub
import os

# Set the OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Set the Pinecone API key
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Set the Google API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Create an instance of ChatOpenAI
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Prompt for the OpenAI Functions agent
prompt = hub.pull("hwchase17/openai-functions-agent")
