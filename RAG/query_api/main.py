from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import os
from dotenv import load_dotenv
from functools import lru_cache
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import uvicorn


class QueryResponse(BaseModel):
    answer: str

class QueryRequest(BaseModel):
    query: str
    llm_model: str
    chunks: List[str]

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Query API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Streamlit frontend, Query API
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define RAG prompt template
prompt_rag = ChatPromptTemplate.from_messages([
    ("system", "Você é um assistente especializado em responder perguntas sobre a legislação municipal de Manaus-AM, Brasil. Utilize os trechos da lei fornecidos para fundamentar suas respostas."),
    ("human", "{query}\n\nContexto relevante da legislação: \n {context}\n\nResposta:"),
])


@lru_cache(maxsize=100)
def get_cached_chain(llm_model: str = 'gpt-4o-mini'):
    """Get or create the RAG chain with caching"""
    # Initialize language model
    model = ChatOpenAI(
        model=llm_model,
        temperature=0.7,
        api_key=os.getenv('OPENAI_API_KEY')
    )
    return prompt_rag | model | StrOutputParser()


@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """
    Process a question using RAG approach with provided chunks.
    
    Args:
        request (QueryRequest): Request containing query and chunks
        
    Returns:
        QueryResponse: The generated answer
    """
    try:
        context = "\n\n".join(request.chunks)
        
        # Get cached chain and generate answer
        chain = get_cached_chain(request.llm_model)
        answer = chain.invoke({
            "query": request.query,
            "context": context
        })
        
        return QueryResponse(answer=answer)
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)