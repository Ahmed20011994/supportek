import os
from langchain.tools.retriever import create_retriever_tool
from openai import OpenAI
from pinecone import Pinecone


# Function to format the output of the language model
def format_lm_output(lm_output):
    last_message_content = str(lm_output.content)
    return {"output": last_message_content}


def get_openai_embeddings(text):
    client = OpenAI()
    response = client.embeddings.create(input=text, model="text-embedding-ada-002")
    return response.data[0].embedding


class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata


class PineconeRetriever:
    def __init__(self, index):
        self.index = index

    async def aget_relevant_documents(self, query, top_k=10, **kwargs):
        query_embedding = get_openai_embeddings(query)
        results = self.index.query(vector=query_embedding, top_k=top_k, include_metadata=True)

        # Process results and handle NoneType for metadata
        documents = []
        for match in results['matches']:
            # Fetch the metadata
            metadata = match['metadata'] if 'metadata' in match else {}
            doc = Document(
                page_content=metadata.get("text", ""),  # Adjust this based on your data structure
                metadata=metadata
            )
            documents.append(doc)

        return documents

    async def ainvoke(self, query, **kwargs):
        return await self.aget_relevant_documents(query, **kwargs)


def create_tools(source):
    # Initialize Pinecone
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

    # Provide the index name
    index_name = str(source['_id'])

    # Connect to the index
    index = pc.Index(index_name)

    # Create a retriever instance
    retriever = PineconeRetriever(index)

    # Create a retrieval tool
    retrieval_tool = create_retriever_tool(
        retriever,
        "client_specific_search",
        "Search tool for client-specific knowledge source.",
    )

    return [retrieval_tool]
