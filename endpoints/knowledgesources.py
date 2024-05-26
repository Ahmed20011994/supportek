import os
from typing import List

import numpy as np
import requests
from bs4 import BeautifulSoup
from bson import ObjectId
from fastapi import APIRouter, HTTPException
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

from database import knowledge_sources_collection
from models import KnowledgeSource

router = APIRouter()


@router.post("/knowledge_sources", response_model=KnowledgeSource)
async def create_knowledge_source(knowledge_source: KnowledgeSource):
    # existing_knowledge_source = knowledge_sources_collection.find_one({"url": knowledge_source.url})
    # if existing_knowledge_source:
    #     raise HTTPException(status_code=400, detail="Knowledge source with the same URL already exists")

    result = knowledge_sources_collection.insert_one(knowledge_source.dict(by_alias=True, exclude={"id"}))
    created_knowledge_source = knowledge_sources_collection.find_one({"_id": result.inserted_id})

    # Extract text content from the URL
    url = knowledge_source.url
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    # Extract all text from the page
    text = soup.get_text(separator="\n")

    # Generate embeddings
    embeddings = get_openai_embeddings(text)

    # Initialize Pinecone
    pc = Pinecone(
        api_key=os.environ.get("PINECONE_API_KEY")
    )

    # Create an index (if not already created)
    index_name = result.inserted_id
    dimension = 1536  # Dimension size for text-embedding-ada-002

    if index_name not in pc.list_indexes():
        pc.create_index(
            name=str(index_name),
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

    # Connect to the index
    index = pc.Index(str(index_name))

    # Upsert vectors into the index
    index.upsert(vectors=[
        (str(result.inserted_id), embeddings, {"text": text})
    ])

    return KnowledgeSource(**created_knowledge_source)


@router.get("/knowledge_sources", response_model=List[KnowledgeSource])
async def get_knowledge_sources():
    knowledge_sources = list(knowledge_sources_collection.find())
    return [KnowledgeSource(**ks) for ks in knowledge_sources]


@router.get("/knowledge_sources/{ks_id}", response_model=KnowledgeSource)
async def get_knowledge_source(ks_id: str):  # Change the type to str
    oid = ObjectId(ks_id)  # Convert the string to ObjectId
    knowledge_source = knowledge_sources_collection.find_one({"_id": oid})
    if not knowledge_source:
        raise HTTPException(status_code=404, detail="Knowledge source not found")
    return KnowledgeSource(**knowledge_source)


@router.put("/knowledge_sources/{ks_id}", response_model=KnowledgeSource)
async def update_knowledge_source(ks_id: str, updated_ks: KnowledgeSource):
    oid = ObjectId(ks_id)
    result = knowledge_sources_collection.update_one({"_id": oid}, {"$set": updated_ks.dict(exclude={"id"})})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Knowledge source not found")
    return updated_ks


@router.delete("/knowledge_sources/{ks_id}")
async def delete_knowledge_source(ks_id: str):
    oid = ObjectId(ks_id)
    result = knowledge_sources_collection.delete_one({"_id": oid})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Knowledge source not found")
    return {"message": "Knowledge source deleted successfully"}


# Function to get OpenAI embeddings
def get_openai_embeddings(text, model="text-embedding-ada-002", max_tokens=8192):
    client = OpenAI()
    chunks = chunk_text(text, max_tokens)
    embeddings = []

    for chunk in chunks:
        response = client.embeddings.create(input=chunk, model=model)
        embeddings.append(response.data[0].embedding)

    # Averaging the embeddings across chunks
    return np.mean(embeddings, axis=0)


def chunk_text(text, max_length):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        if current_length + len(word) > max_length:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
        else:
            current_chunk.append(word)
            current_length += len(word)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
