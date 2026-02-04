from typing import TypedDict, List
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

import json
import os

class SQLState(TypedDict):
    user_query: str
    sql_query: str
    selected_tables: List[str]
    filtered_kb: dict


def get_table_names() -> List[str]:
    return ["dim_ticket"]
table_names = get_table_names()


### Table Descriptions
table_descriptions = {
    "dim_ticket": "Tabela dimensional contendo informações consolidadas de tickets, incluindo dados de identificação, demanda, responsáveis, datas relevantes, status e características específicas como subtickets e vínculos.",
}
def get_table_description(table_name: str) -> str:
    return table_descriptions.get(table_name.lower(), f"-- Tabela '{table_name}' não encontrada.")

def get_all_table_descriptions() -> str:
    return '\n'.join([f"- {table_name}: {desc}" for table_name, desc in table_descriptions.items()])

table_schemas = {
    "dim_ticket": """
        CREATE TABLE dim_ticket (
            ID_DIM_TICKET NUMBER,
            TICKET_PRINCIPAL NUMBER,
            RESUMO_TICKET VARCHAR2(4000 BYTE),
            RESUMO_TICKET_PRINCIPAL VARCHAR2(4000 BYTE),
            DESCRICAO VARCHAR2(4000 BYTE),
            IDENTIFICACAO_DEMANDA VARCHAR2(4000 BYTE),
            TEMA VARCHAR2(4000 BYTE),
            SERVICO VARCHAR2(80 BYTE),
            DATA_ULTIMA_MODIFICACAO DATE,
            SITUACAO VARCHAR2(40 BYTE),
            FLAG_FECHAR NUMBER DEFAULT 0, 
            FLAG_POSSUI_SUBTICKET NUMBER DEFAULT 0, 
            FLAG_PRINCIPAL NUMBER DEFAULT 0, 
            DEMANDANTE VARCHAR2(180 BYTE), 
            CRIADOR VARCHAR2(180 BYTE), 
            DATA_ABERTURA DATE, 
            DATA_FECHAMENTO DATE, 
            QTD_SUBTICKETS NUMBER DEFAULT 0
        );
    """
}
def get_table_schema(table_name: str) -> str:
    return table_schemas.get(table_name.lower(), f"-- Esquema para a tabela '{table_name}' não encontrado.")

table_column_descriptions = {
    "dim_ticket": """
            --ID_DIM_TICKET: Identificador único da dimensão de ticket. Chave primária gerada para uso no Data Warehouse.
            --TICKET_PRINCIPAL: Identificador único do ticket principal ao qual o registro está associado. Quando não houver o valor será zero.
            --RESUMO_TICKET: Resumo ou título descritivo do ticket atual.
            --RESUMO_TICKET_PRINCIPAL: Resumo ou título do ticket principal vinculado a este ticket.
            --DESCRICAO: Descrição completa do ticket atual.
            --IDENTIFICACAO_DEMANDA: Texto descritivo de identificação da demanda.
            --TEMA: Tema ou categoria principal associada ao ticket.
            --SERVICO: Categoria de serviço relacionado ao ticket, utilizado para classificação funcional da demanda.
            --DATA_ULTIMA_MODIFICACAO: Data da última modificação registrada no ticket.
            --SITUACAO: Situação atual do ticket.
            --FLAG_FECHAR: Indicador binário que identifica se o ticket deveria estar fechado (0 = não, 1 = sim).
            --FLAG_POSSUI_SUBTICKET: Indicador binário que informa se o ticket possui subtickets associados (0 = não, 1 = sim).
            --FLAG_PRINCIPAL: Indica se o ticket é o ticket principal dentro de uma hierarquia (0 = não, 1 = sim).
            --DEMANDANTE: Nome do demandante que solicitou a abertura do ticket.
            --CRIADOR: Nome ou identificação do usuário que criou ou registrou o ticket no sistema.
            --DATA_ABERTURA: Data de abertura do ticket.
            --DATA_FECHAMENTO: Data de fechamento do ticket, quando aplicável.
            --QTD_SUBTICKETS: Quantidade total de subtickets vinculados ao ticket principal.
        """,
}
def get_column_descriptions(table_name: str) -> str:
    return table_column_descriptions.get(table_name.lower(), f"-- Descrições de colunas para a tabela '{table_name}' não encontradas.")


load_dotenv()

# Initialize Language Model
model = ChatOpenAI(
    model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
    temperature=0.7,
        api_key=os.getenv('OPENAI_API_KEY')
)

# Define Prompt Templates
router_template = ChatPromptTemplate.from_messages([
    ("system", """Você é um router especializado em sistemas Text-to-SQL. Dado uma consulta do usuário e as descrições das tabelas, retorne como única saída uma lista com os nomes das tabelas que são relevantes para responder a consulta."""),
    ("user", """
        Considere um banco de dados dimensional com as seguintes tabelas:
        {table_descriptions}
     
        Instruções:
            1. Analise cuidadosamente a consulta do usuário e divida se possível em subconsultas.
            2. Para cada subconsultas, verifique as descrições das tabelas e identifique as tabelas relevantes necessárias para responder ao usuário.
            3. Gere lista com os nomes das tabelas selecionadas.
            4. A saída deve ser um array, contendo somente os nomes das tabelas selecionadas. Exemplo: ["tabela1", "tabela2"]
        
        Consulta do usuário: {user_query}
    """
    ),
])

sqlgen_template = ChatPromptTemplate.from_messages([
    ("system", """Você é um especialista em SQL para banco de dados Oracle. Com base na consulta do usuário e nas tabelas fornecidas, gere a consulta SQL correta para obter os dados solicitados. Utilize a sintaxe SQL adequada para Oracle."""),
    ("user", """
        Considere o seguinte esquema do banco de dados:
        {filtered_kb} 
        
        Instruções:
            1. Analise a consulta do usuário e as tabelas fornecidas.
            2. Gere a consulta SQL correta para obter os dados solicitados.
            3. Certifique-se de que a consulta esteja em conformidade com a sintaxe SQL do Oracle.
            4. Sua resposta deve conter apenas a consulta SQL, sem explicações adicionais.
        
        Consulta do usuário: {user_query}
        """
    ),
])


# Define Nodes

# Router Agent Node
def router_agent(state: SQLState) -> SQLState:
    router_chain = router_template | model
    
    response = router_chain.invoke({
        "table_descriptions": json.dumps(table_descriptions, indent=2),
        "user_query": state["user_query"]
    })

    selected_tables = str(response.content)
    selected_tables = json.loads(selected_tables)
    state["selected_tables"] = selected_tables
    
    filtered_kb = {
        table_name: {
            'description': get_table_description(table_name),
            'schema': get_table_schema(table_name),
            'column_descriptions': get_column_descriptions(table_name)
        }
        for table_name in selected_tables
    }
    state["filtered_kb"] = filtered_kb
    
    return state

# SQL Generation Node
def sql_generation_agent(state: SQLState) -> SQLState:
    sql_chain = sqlgen_template | model
    
    response = sql_chain.invoke({
        "filtered_kb": json.dumps(state["filtered_kb"], indent=2),
        "user_query": state["user_query"]
    })
    
    state["sql_query"] = response
    
    return state

# Build State Graph
graph = StateGraph(SQLState)

graph.add_node("RouterAgent", router_agent)
graph.add_node("SQLGenerationAgent", sql_generation_agent)

graph.add_edge(START, "RouterAgent")
graph.add_edge("RouterAgent", "SQLGenerationAgent")
graph.add_edge("SQLGenerationAgent", END)

workflow = graph.compile()

# Test the workflow with a sample user query
user_query = "Quais são os tickets com status 'Em Andamento/Aberto', que possuam ao menos um subticket, e que seus subtickets estejam todos com status 'Fechado'?"
test_state: SQLState = {
    "user_query": user_query,
    "sql_query": "",
    "selected_tables": [],
    "filtered_kb": {}
}

# Execute workflow
final_state = workflow.invoke(test_state)
print("User query", final_state["user_query"])
print("\n\nSelected tables", final_state["selected_tables"])
print("\n\nFiltered KB", final_state["filtered_kb"])
print("\n\nSQL query", final_state["sql_query"])
# print("Final SQL Query:", final_state.content)