from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import os

load_dotenv()

model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    api_key=os.getenv('OPENAI_API_KEY')
)   

embeddings = OpenAIEmbeddings()

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

chunks = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
).split_documents(corpus)

vectorstore = FAISS.from_documents(chunks, embeddings).as_retriever(search_kwargs={"k": 3})

prompt_rag = ChatPromptTemplate.from_messages(
    [
        ("system", "Você é um assistente especializado em responder perguntas sobre a legislação municipal de Manaus-AM, Brasil. Utilize os trechos da lei fornecidos para fundamentar suas respostas."),
        ("human", "{query}\n\nContexto relevante da legislação: \n {context}\n\nResposta:"),
    ]
)

chain = prompt_rag | model | StrOutputParser()

def rag_query(query: str) -> str:
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

print(rag_query("Quais são as alíquotas do ISS para serviços de consultoria?"))