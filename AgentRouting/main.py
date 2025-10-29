from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig


import os
import asyncio


class AgentRoute(TypedDict):
    route: Literal["aventura", "praia", "gastronomia"]


class AgentState(TypedDict):
    query: str
    destino: AgentRoute
    resposta: str


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

chain_aventura = prompt_agent_aventura | model | StrOutputParser()
chain_praia = prompt_agent_beach | model | StrOutputParser()
chain_gastronomia = prompt_agent_gastronomia | model | StrOutputParser()

agent_router = prompt_routing | model.with_structured_output(AgentRoute)

async def node_routing(state: AgentState, config=RunnableConfig):
    return {
        'destino': await agent_router.ainvoke({'query': state['query']}, config=config)
    }

async def node_praia(state: AgentState, config=RunnableConfig):
    return {
        'resposta': await chain_praia.ainvoke({'query': state['query']}, config=config)
    }

async def node_aventura(state: AgentState, config=RunnableConfig):
    return {
        'resposta': await chain_aventura.ainvoke({'query': state['query']}, config=config)
    }

async def node_gastronomia(state: AgentState, config=RunnableConfig):
    return {
        'resposta': await chain_gastronomia.ainvoke({'query': state['query']}, config=config)
    }


def node_select(state: AgentState) -> Literal['praia', 'aventura', 'gastronomia']:
    return state['destino']['route']


grafo = StateGraph(AgentState)
grafo.add_node('rotear', node_routing, outputs=['destino'])
grafo.add_node('aventura', node_aventura, outputs=['resposta']) 
grafo.add_node('gastronomia', node_gastronomia, outputs=['resposta'])
grafo.add_node('praia', node_praia, outputs=['resposta'])

grafo.add_edge(START, 'rotear')
grafo.add_conditional_edges('rotear', node_select)
grafo.add_edge('aventura', END)
grafo.add_edge('gastronomia', END)
grafo.add_edge('praia', END)

app = grafo.compile()

async def main():
    query = input('Descreva o tipo de viagem que pretende fazer: ')
    resposta = await app.ainvoke(
        {'query': query}
    )
    print(resposta['resposta'])
    
asyncio.run(main())