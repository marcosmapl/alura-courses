from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import TypedDict, Literal

import os


class AgentRoutes(TypedDict):
    route: Literal["aventura", "praia", "gastronomia"]


load_dotenv()


model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.5,
    api_key=os.getenv('OPENAI_API_KEY')
)

prompt_agent_beach = ChatPromptTemplate.from_messages(
    [
        ("system", "Você é um agente de viagens especializado em destinos de praia no Brasil. Apresente-se como Sr. Praias."),
        ("human", "{query}"),
    ]
)


prompt_agent_aventura = ChatPromptTemplate.from_messages(
    [
        ("system", "Você é um agente de viagens especializado em esportes radicais, trilhas e aventura do Brasil. Apresente-se como Sr. Aventura."),
        ("human", "{query}"),
    ]
)


prompt_agent_gastronomia = ChatPromptTemplate.from_messages(
    [
        ("system", "Você é um agente de viagens especializado em gastronomia do Brasil. Apresente-se como Sr. Gastronomia."),
        ("human", "{query}"),
    ]
)

prompt_routing = ChatPromptTemplate.from_messages(
    [
        ("system", "Você é um roteador de agentes de viagem. Se a consulta for sobre destinos de praia, responda com a palavra 'praia', se for sobre esportes radicais ou aventura, responda com a palavra 'aventura', caso contrário, responda com a palavra 'gastronomia'."),
        ("human", "{query}"),
    ]
)

chain_praias = prompt_agent_beach | model | StrOutputParser()
chain_gastronomia = prompt_agent_gastronomia | model | StrOutputParser()
chain_aventura = prompt_agent_aventura | model | StrOutputParser()

agent_router = prompt_routing | model.with_structured_output(AgentRoutes)


def seleciona_agente(query: str, router) -> str:
    response = router.invoke(
        {"query": query}
    )['route']
    
    print(response)
    
    if str(response).lower() == 'praia':
        return chain_praias
    elif str(response).lower() == 'aventura':
        return chain_aventura
    
    return chain_gastronomia


query = input('Qual o seu objetivo de viagem? ')
agent = seleciona_agente(query, agent_router)
response = agent.invoke(
    {"query": query}
)   

print(response)
