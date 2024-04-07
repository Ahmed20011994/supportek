from fastapi import FastAPI

from endpoints import langchain, knowledgesources

app = FastAPI(title="Supportek LLM Server", version="1.0", description="LLM based API server")

app.include_router(langchain.router, prefix="/llm", tags=["LangChain"])
app.include_router(knowledgesources.router, prefix="/llm", tags=["Knowledge Sources"])


@app.get("/llm/health")
def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
