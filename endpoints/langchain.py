import os
import re

import requests
from bson import ObjectId
from bson.errors import InvalidId
from fastapi import APIRouter
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langdetect import detect

from config import llm, prompt
from core_logic import format_lm_output, create_tools
from database import knowledge_sources_collection
from models import HumanMessage, Input, Output

router = APIRouter()


@router.post("/langchain", response_model=Output)
async def handle_request(user_input: Input):
    knowledge_source_id: str = user_input.knowledge_source_id
    tools = []

    try:
        source = knowledge_sources_collection.find_one({"_id": ObjectId(knowledge_source_id)})
        if source:
            tools = create_tools(source)
    except InvalidId:
        # If the knowledge_source_id is invalid, proceed without tools
        pass

    detected_language = detect_language(user_input.input)

    if detected_language != 'en':
        user_input.input = google_translate(user_input.input, 'en')

    # Prepare the input message and chat history
    input_message = HumanMessage(content=user_input.input)
    chat_history = user_input.chat_history

    # Combine the input message with the chat history for the context
    context = chat_history + [input_message]

    # Check if there are any tools to use with the agent
    if tools:
        # Use the agent with the retrieval tool
        agent = create_openai_functions_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        agent_input = {"input": context, "context": context}
        response = await agent_executor.ainvoke(agent_input)
        match = re.search(r'(AssistantMessage|AIMessage)\(content=\'(.*)\'\)', response.get("output"))
        if match:
            content = match.group(2)
        else:
            content = response.get("output")
        output = content
    else:
        # Directly use the LLM without any tools
        lm_output = await llm.ainvoke(context)  # Pass the context including chat history
        output = format_lm_output(lm_output).get("output")

    if detected_language != 'en':
        output = google_translate(output, detected_language)

    return {"output": output}


def detect_language(text):
    # Detect the language of the text
    detected_language = detect(text)
    return detected_language


def google_translate(text, target_language):
    url = "https://translation.googleapis.com/language/translate/v2"
    data = {
        'q': text,
        'target': target_language,
        'format': 'text'
    }
    headers = {
        'Content-Type': 'application/json',
        'Accept-Charset': 'UTF-8',
        'X-Goog-Api-Key': os.environ.get("GOOGLE_API_KEY")
    }

    # Making the POST request
    response = requests.post(url, params=data, headers=headers)

    # Checking if the request was successful
    if response.status_code == 200:
        # Parsing the response
        result = response.json()
        translations = result.get('data', {}).get('translations', [])
        if translations:
            return translations[0].get('translatedText')
    else:
        return f"Failed to translate text: {response.text}"

