from typing import List

from bson import ObjectId
from fastapi import APIRouter, HTTPException

from database import knowledge_sources_collection
from models import KnowledgeSource

router = APIRouter()


@router.post("/", response_model=KnowledgeSource)
async def create_knowledge_source(knowledge_source: KnowledgeSource):
    existing_knowledge_source = knowledge_sources_collection.find_one({"url": knowledge_source.url})
    if existing_knowledge_source:
        raise HTTPException(status_code=400, detail="Knowledge source with the same URL already exists")
    result = knowledge_sources_collection.insert_one(knowledge_source.dict(by_alias=True, exclude={"id"}))
    created_knowledge_source = knowledge_sources_collection.find_one({"_id": result.inserted_id})
    return KnowledgeSource(**created_knowledge_source)


@router.get("/", response_model=List[KnowledgeSource])
async def get_knowledge_sources():
    knowledge_sources = list(knowledge_sources_collection.find())
    return [KnowledgeSource(**ks) for ks in knowledge_sources]


@router.get("/{ks_id}", response_model=KnowledgeSource)
async def get_knowledge_source(ks_id: str):  # Change the type to str
    oid = ObjectId(ks_id)  # Convert the string to ObjectId
    knowledge_source = knowledge_sources_collection.find_one({"_id": oid})
    if not knowledge_source:
        raise HTTPException(status_code=404, detail="Knowledge source not found")
    return KnowledgeSource(**knowledge_source)


@router.put("/{ks_id}", response_model=KnowledgeSource)
async def update_knowledge_source(ks_id: str, updated_ks: KnowledgeSource):
    oid = ObjectId(ks_id)
    result = knowledge_sources_collection.update_one({"_id": oid}, {"$set": updated_ks.dict(exclude={"id"})})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Knowledge source not found")
    return updated_ks


@router.delete("/{ks_id}")
async def delete_knowledge_source(ks_id: str):
    oid = ObjectId(ks_id)
    result = knowledge_sources_collection.delete_one({"_id": oid})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Knowledge source not found")
    return {"message": "Knowledge source deleted successfully"}
