# from fastapi import FastAPI, Request
# from pydantic import BaseModel
# from agent import ask_question

# app = FastAPI()

# class QueryRequest(BaseModel):
#     q: str

# @app.post("/query")
# def query(request: QueryRequest):
#     answer = ask_question(request.q)
#     return {"answer": answer}


import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from agent import LLMGraphQLAgent

load_dotenv()


app = FastAPI()

class QueryRequest(BaseModel):
    q: str

class QueryResponse(BaseModel):
    answer: str

openai_api_key = os.getenv("OPENAI_API_KEY")
graphql_url = os.getenv("GRAPHQL_API_URL")

agent = LLMGraphQLAgent(openai_api_key, graphql_url)

@app.get("/")
async def root():
    return {"message": "LLM GraphQL Agent is running"}


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    answer = agent.query(request.q)
        
    return QueryResponse(answer=answer)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)