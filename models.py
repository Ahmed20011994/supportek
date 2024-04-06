from typing import List, Union, Optional

from bson import ObjectId
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage


class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if isinstance(v, ObjectId):
            return str(v)
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return str(ObjectId(v))

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type="string")


class KnowledgeSource(BaseModel):
    id: Optional[PyObjectId] = Field(None, alias="_id")
    url: str
    description: str


class Input(BaseModel):
    input: str
    knowledge_source_id: str
    chat_history: List[Union[AIMessage, HumanMessage]] = Field(
        ...,
        extra={"widget": {"type": "chat", "input": "location"}},
    )


class Output(BaseModel):
    output: str
