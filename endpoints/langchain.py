from bson import ObjectId
from bson.errors import InvalidId
from fastapi import APIRouter

from config import llm, prompt
from core_logic import format_lm_output, create_tools
from database import knowledge_sources_collection
from models import HumanMessage, Input, Output

router = APIRouter()


@router.post("/", response_model=Output)
async def handle_request(user_input: Input):
    knowledge_source_id: str = user_input.knowledge_source_id
    tools = []

    try:
        source = knowledge_sources_collection.find_one({"_id": ObjectId(knowledge_source_id)})
        if source:
            tools = create_tools(source['url'])
    except InvalidId:
        # If the knowledge_source_id is invalid, proceed without tools
        pass

    # Prepare the input message
    input_message = HumanMessage(content=user_input.input)

    # Check if there are any tools to use with the agent
    if tools:
        # Use the agent with the retrieval tool
        from langchain.agents import AgentExecutor, create_openai_functions_agent
        agent = create_openai_functions_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        agent_input = {"input": [input_message], "context": user_input.chat_history}
        response = await agent_executor.ainvoke(agent_input)
        output = response.get("output")
    else:
        # Directly use the LLM without any tools
        lm_output = await llm.ainvoke([input_message])
        output = format_lm_output(lm_output).get("output")

    return {"output": output}
