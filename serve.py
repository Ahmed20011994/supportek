import os
from typing import List, Union
from fastapi import FastAPI, Depends
from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents import create_openai_functions_agent
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import Tool  # Import the Tool base class
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langserve import add_routes

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# Placeholder tool class
class PlaceholderTool(Tool):
    def __init__(self, name="Placeholder Tool", func=None, description="This is a placeholder tool."):
        super().__init__(name, func, description)

    def run(self, *args, **kwargs):
        return "This is a placeholder tool."


# Function to create tools based on clientId and chatbotId
def create_tools(client_id: str, chatbot_id: str):
    tools = []
    retriever = load_retriever(client_id, chatbot_id)
    if retriever:
        retriever_tool = create_retriever_tool(
            retriever,
            "custom_search",
            "Search for information based on clientId and chatbotId.",
        )
        tools.append(retriever_tool)
    return tools


# 1. Load Retriever based on clientId and chatbotId
def load_retriever(client_id: str, chatbot_id: str):
    if client_id == "langsmith" and chatbot_id == "chatbot1":
        url = "https://docs.smith.langchain.com/user_guide"
    elif client_id == "langsmith" and chatbot_id == "chatbot2":
        url = "https://docs.smith.langchain.com/user_guide"
    else:
        return None  # Return None if no matching URL is found

    loader = WebBaseLoader(url)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    vector = FAISS.from_documents(documents, embeddings)
    retriever = vector.as_retriever()
    return retriever


# 3. Create Agent with initial tools
initial_tools = create_tools("", "")
if not initial_tools:
    initial_tools.append(PlaceholderTool())
prompt = hub.pull("hwchase17/openai-functions-agent")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
agent = create_openai_functions_agent(llm, initial_tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=initial_tools, verbose=True)

# 4. App definition
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API server using LangChain's Runnable interfaces",
)


# 5. Adding chain route
class Input(BaseModel):
    clientId: str
    chatbotId: str
    input: str
    chat_history: List[Union[AIMessage, HumanMessage]] = Field(
        ...,
        extra={"widget": {"type": "chat", "input": "location"}},
    )


class Output(BaseModel):
    output: str


def get_agent_executor(get_agent_executor_input: Input = Depends()) -> AgentExecutor:
    tools = create_tools(get_agent_executor_input.clientId, get_agent_executor_input.chatbotId)
    agent_executor.tools = tools
    agent_executor.agent.tools = tools
    return agent_executor


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
