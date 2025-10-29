from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import os

# Load environment variables from .env file
load_dotenv()

# Initialize the language model
model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    api_key=os.getenv('OPENAI_API_KEY')
)   

# Initialize embeddings model
embeddings = OpenAIEmbeddings()

# Load PDF files and create a corpus
pdf_files = [
    'RAG/data/Lei-ordinaria-2833-2021-Manaus-AM-consolidada-[20-08-2024].pdf',
    'RAG/data/Lei-ordinaria-459-1998-Manaus-AM-consolidada-[26-12-2019].pdf',
    'RAG/data/Lei-ordinaria-1628-2011-Manaus-AM-consolidada-[20-12-2024].pdf',
]

corpus = sum(
    [
        PyPDFLoader(pdf).load() for pdf in pdf_files
    ], []
)

# Split documents into chunks
chunks = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
).split_documents(corpus)

# Create a FAISS vector store from the document chunks
vectorstore = FAISS.from_documents(chunks, embeddings).as_retriever(search_kwargs={"k": 3})

# Define the RAG prompt template
prompt_rag = ChatPromptTemplate.from_messages(
    [
        ("system", "Você é um assistente especializado em responder perguntas sobre a legislação municipal de Manaus-AM, Brasil. Utilize os trechos da lei fornecidos para fundamentar suas respostas."),
        ("human", "{query}\n\nContexto relevante da legislação: \n {context}\n\nResposta:"),
    ]
)

# Create the RAG chain
chain = prompt_rag | model | StrOutputParser()

# Function to perform RAG query
def rag_query(query: str) -> str:
    """
    Execute a retrieval-augmented generation (RAG) query by fetching relevant document
    chunks from a vector store, assembling them into a context, and invoking a
    generation chain to produce an answer.

    Parameters
    ----------
    query : str
        The user's natural-language query to answer.

    Returns
    -------
    str
        The response produced by the generation chain. This is the value returned
        from chain.invoke for the given query and assembled context.

    Behavior / Side effects
    -----------------------
    - Calls vectorstore.invoke(query) to retrieve relevant chunks; each chunk is
      expected to have a `page_content` attribute.
    - Prints the raw `relevant_chunks` value to stdout for debugging.
    - Joins retrieved chunk contents with two newline characters to form the
      context passed to the chain.
    - Calls chain.invoke({"query": query, "context": context}, config={"verbose": True})
      and returns its result.
    - May perform I/O or network calls depending on the implementations of
      `vectorstore` and `chain`.

    Raises
    ------
    AnyExceptionFromVectorstore
        Exceptions raised by vectorstore.invoke (e.g., connection or retrieval errors)
        are propagated.
    AnyExceptionFromChain
        Exceptions raised by chain.invoke (e.g., model or generation errors) are propagated.
    AttributeError
        If retrieved chunks do not have the expected `page_content` attribute.

    Example
    -------
    >>> resp = rag_query("What is retrieval-augmented generation?")
    """
    relevant_chunks = vectorstore.invoke(query)
    print(relevant_chunks)
    context = "\n\n".join([relevant_chunk.page_content for relevant_chunk in relevant_chunks])
    resposta = chain.invoke(
        {
            "query": query,
            "context": context
        },
        config={
            "verbose": True
        }
    )
    return resposta

query = input('Digite sua pergunta sobre a legislação tributária de Manaus-AM: ')
response = rag_query(query)
print(response)