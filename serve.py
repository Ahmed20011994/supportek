import os
from typing import List, Union

from fastapi import FastAPI
from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents import create_openai_functions_agent
from langchain.pydantic_v1 import Field
from langchain.tools import retriever
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langserve import add_routes
from pydantic import BaseModel

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


# Define RunnableTool class
class RunnableTool:
    def __init__(self, name, description, runnable, input_schema, output_schema):
        self.name = name
        self.description = description
        self.runnable = runnable
        self.input_schema = input_schema
        self.output_schema = output_schema

    def run(self, *args, **kwargs):
        return self.runnable(*args, **kwargs)


# Define TextQueryInput and TextQueryOutput
class TextQueryInput(BaseModel):
    query: str
    clientId: str
    chatbotId: str


class TextQueryOutput(BaseModel):
    result: str


# 1. Create Tools
def create_retriever_tool(modified_retriever, tool_name: str, description: str):
    """
    Create a retriever tool with hardcoded knowledge sources based on client_id and chatbot_id.
    """

    def retrieve(client_id, chatbot_id, query):
        url_mapping = {
            ("langsmith", "chatbot1"): "https://docs.smith.langchain.com/user_guide"
        }
        url = url_mapping.get((client_id, chatbot_id))
        if url:
            loader = WebBaseLoader(url)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter()
            documents = text_splitter.split_documents(docs)
            embeddings = OpenAIEmbeddings()
            vector = FAISS.from_documents(documents, embeddings)
            return retriever.retrieve(query, source=vector.as_retriever())
        else:
            return None

    return RunnableTool(
        name=tool_name,
        description=description,
        runnable=retrieve,
        input_schema=TextQueryInput,
        output_schema=TextQueryOutput,
    )


# 2. Load Retriever
retriever_tool = create_retriever_tool(
    retriever,
    "langsmith_search",
    "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!",
)
search = TavilySearchResults()
tools = [retriever_tool, search]

# 3. Create Agent
prompt = hub.pull("hwchase17/openai-functions-agent")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 4. App definition
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API server using LangChain's Runnable interfaces",
)


# 5. Adding chain route

# We need to add these input/output schemas because the current AgentExecutor
# is lacking in schemas.

class Input(BaseModel):
    input: str
    chat_history: List[Union[AIMessage, HumanMessage]] = Field(
        ...,
        extra={"widget": {"type": "chat", "input": "location"}},
    )
    clientId: str
    chatbotId: str


class Output(BaseModel):
    output: str


add_routes(
    app,
    agent_executor.with_types(input_type=Input, output_type=Output),
    path="/supportek",
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
