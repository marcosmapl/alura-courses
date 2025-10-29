from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

import os
import uuid


def historico_chat(sessao_id: str, memoria: dict) -> str:
    if sessao_id not in memoria:
        memoria[sessao_id] = InMemoryChatMessageHistory()
    return memoria[sessao_id]


load_dotenv()

model = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    api_key=os.getenv('OPENAI_API_KEY')
)

prompt_sugestao = ChatPromptTemplate.from_messages(
    [
        ("system", "Você é um guia de viagem especialista em destinos de viagem no Brasil. Apresente-se como Sr. Destinos."),
        ("placeholder", "{history}"),
        ("human", "{query}"),
    ]
)

chain = prompt_sugestao | model | StrOutputParser()

memoria = {}
sessao = str(uuid.uuid4())

chain_memory = RunnableWithMessageHistory(
    runnable=chain,
    get_session_history=lambda _: historico_chat(sessao, memoria),
    input_messages_key="query",
    history_messages_key="history"
)

perguntas = [
    "Quero visitar um lugar no Brasil, famoso por praias e cultura. Pode sugerir um destino?",
    "Qual a melhor época do ano para viajar para esse destino?",
]

for pergunta in perguntas:
    resposta = chain_memory.invoke(
        {
            "query": pergunta
        },
        config={
            "verbose": True,
            "session_id": sessao
        }
    )
    print(f"Usuário: {pergunta}")
    print(f"LLM: {resposta}\n")