import os
from typing import List, Union
from fastapi import FastAPI, Depends
from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents import create_openai_functions_agent
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


# 1. Load Retriever based on clientId and chatbotId
def load_retriever(client_id: str, chatbot_id: str):
    if client_id == "langsmith" and chatbot_id == "chatbot1":
        loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter()
        documents = text_splitter.split_documents(docs)
        embeddings = OpenAIEmbeddings()
        vector = FAISS.from_documents(documents, embeddings)
        retriever = vector.as_retriever()
        return retriever
    return None


# 2. Create Tools
search = TavilySearchResults()

# 3. Create Agent
prompt = hub.pull("hwchase17/openai-functions-agent")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
agent = create_openai_functions_agent(llm, [], prompt)  # Initialize with empty tools
agent_executor = AgentExecutor(agent=agent, tools=[], verbose=True)  # Initialize with empty tools

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
    retriever = load_retriever(get_agent_executor_input.clientId, get_agent_executor_input.chatbotId)
    tools = [search]
    if retriever:
        retriever_tool = create_retriever_tool(
            retriever,
            "langsmith_search",
            "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!",
        )
        tools.append(retriever_tool)
    agent_executor.tools = tools
    agent_executor.agent.tools = tools
    return agent_executor


@app.post("/supportek", response_model=Output)
async def supportek_endpoint(support_input: Input, support_agent_executor: AgentExecutor = Depends(get_agent_executor)):
    return await support_agent_executor.run(support_input)


@app.get("/health")
def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
