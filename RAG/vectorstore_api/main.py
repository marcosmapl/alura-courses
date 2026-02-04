from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

import uvicorn
import corpus

class ChunkModel(BaseModel):
    page_content: str
    metadata: Dict[str, Any] = {}

class SearchResponse(BaseModel):
    chunks: List[ChunkModel]

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="VectorStore API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Streamlit frontend, Query API
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize embeddings model
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

# Load documents and create corpus
corpus = corpus.build_corpus()

# Create FAISS vector store
vectorstore = FAISS.from_documents(corpus, embeddings)

# Configure caching
from functools import lru_cache
@lru_cache(maxsize=1000)
def cached_similarity_search(query: str, k: int = 3):
    return vectorstore.similarity_search(query, k=k)


@app.get("/search", response_model=SearchResponse)
async def search_chunks(query: str, k: int = 3):
    """
    Search for relevant document chunks based on the query.
    
    Args:
        query (str): The search query
        
    Returns:
        SearchResponse: List of relevant text chunks
    """
    try:
        # Get relevant documents from vector store using cache
        relevant_docs = cached_similarity_search(query, k)
        
        # Extract page content and metadata from documents
        response_chunks: List[ChunkModel] = []
        for doc in relevant_docs:
            print("Document content:", doc)
            metadata = getattr(doc, 'metadata', None) or {}
            response_chunks.append(ChunkModel(page_content=doc.page_content, metadata=metadata))

        return SearchResponse(chunks=response_chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)