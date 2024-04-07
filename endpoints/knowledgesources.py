from typing import List

from bson import ObjectId
from fastapi import APIRouter, HTTPException
from langchain_openai import OpenAIEmbeddings

from core_logic import load_documents
from database import knowledge_sources_collection
from database import vectors_collection
from models import KnowledgeSource

router = APIRouter()


@router.post("/knowledge_sources", response_model=KnowledgeSource)
async def create_knowledge_source(knowledge_source: KnowledgeSource):
    existing_knowledge_source = knowledge_sources_collection.find_one({"url": knowledge_source.url})
    if existing_knowledge_source:
        raise HTTPException(status_code=400, detail="Knowledge source with the same URL already exists")

    result = knowledge_sources_collection.insert_one(knowledge_source.dict(by_alias=True, exclude={"id"}))
    created_knowledge_source = knowledge_sources_collection.find_one({"_id": result.inserted_id})

    # Load documents from the provided URL
    documents = load_documents(knowledge_source.url)

    # Generate embeddings for the documents
    embeddings_model = OpenAIEmbeddings()
    vectors = embeddings_model.embed_documents(documents)

    for vector in vectors:
        vector_dict = {"knowledge_source_id": result.inserted_id, "vector": vector}
        vectors_collection.insert_one(vector_dict)

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
