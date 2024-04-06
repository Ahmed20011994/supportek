from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool


# Function to load documents from knowledge source
def load_documents(source):
    loader = WebBaseLoader(source)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)
    return documents


# Function to format the output of the language model
def format_lm_output(lm_output):
    last_message_content = str(lm_output.content)
    return {"output": last_message_content}


# Function to create tools for retrieval
def create_tools(source):
    documents = load_documents(source)
    embeddings = OpenAIEmbeddings()
    vector = FAISS.from_documents(documents, embeddings)
    retriever = vector.as_retriever()
    retrieval_tool = create_retriever_tool(
        retriever,
        "client_specific_search",
        "Search tool for client-specific knowledge source.",
    )
    return [retrieval_tool]
