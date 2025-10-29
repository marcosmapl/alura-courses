from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

import os
import uuid


# function to get or create chat history for a session
def historico_chat(sessao_id: str, memoria: dict) -> str:
    """
    Retorna (ou cria) o histórico de chat associado a uma sessão.

    Parâmetros:
        sessao_id (str): Identificador único da sessão de chat.
        memoria (dict): Dicionário que mapeia ids de sessão para objetos de histórico de mensagens.
                       Espera-se que armazene instâncias de InMemoryChatMessageHistory. Se sessao_id
                       não existir em memoria, uma nova InMemoryChatMessageHistory() será criada e
                       atribuída a essa chave (mutação de memoria).

    Retorno:
        InMemoryChatMessageHistory: O objeto de histórico de chat associado a sessao_id.

    Exceções:
        TypeError: Se memoria não suportar operações de indexação/atribuição como um mapping.

    Observações:
        - A função pressupõe que InMemoryChatMessageHistory esteja disponível no escopo.
        - Não é segura para uso concorrente; chamadas simultâneas podem resultar na criação de múltiplas
          instâncias para a mesma sessão.
    """
    if sessao_id not in memoria:
        memoria[sessao_id] = InMemoryChatMessageHistory()
    return memoria[sessao_id]

# Load environment variables from a .env file
load_dotenv()

# Initialize the ChatOpenAI model
model = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    api_key=os.getenv('OPENAI_API_KEY')
)

# Define the chat prompt template
prompt_sugestao = ChatPromptTemplate.from_messages(
    [
        ("system", "Você é um guia de viagem especialista em destinos de viagem no Brasil. Apresente-se como Sr. Destinos."),
        ("placeholder", "{history}"),
        ("human", "{query}"),
    ]
)

# Create the processing chain
chain = prompt_sugestao | model | StrOutputParser()

# Initialize memory and session
memoria = {}
sessao = str(uuid.uuid4())

# Wrap the chain with message history handling
chain_memory = RunnableWithMessageHistory(
    runnable=chain,
    get_session_history=lambda _: historico_chat(sessao, memoria),
    input_messages_key="query",
    history_messages_key="history"
)

# Example questions to interact with the chat
perguntas = [
    "Quero visitar um lugar no Brasil, famoso por praias e cultura. Pode sugerir um destino?",
    "Qual a melhor época do ano para viajar para esse destino?",
]

# Process each question and print the responses
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