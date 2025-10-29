# from openai import OpenAI
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.globals import set_debug
from pydantic import BaseModel, Field

import os


class CNAE(BaseModel):
    cnae: str = Field("O código CNAE (classificação nacional de atividades econômicas) recomendado.")
    motivo: str = Field("O motivo pelo qual o código CNAE foi recomendado.")


class Semelhante(BaseModel):
    cnae: str = Field("O código CNAE (classificação nacional de atividades econômicas) recomendado.")
    semelhantes: list[str] = Field("Uma lista com os nomes outros códigos CNAE semelhantes.")


set_debug(True)
load_dotenv()

# Initialize the JSON output parser
parser_cnae = JsonOutputParser(pydantic_object=CNAE)
parser_semelhante = JsonOutputParser(pydantic_object=Semelhante)

# Define the prompt template
prompt_cnae = PromptTemplate(
    template="""
    Sugira um código CNAE (classificação nacionale de atividades econômicas) para uma empresa que executa as atividades de: {atividades}.
    {output_format}
    """,
    input_variables=["atividades"],
    partial_variables={'output_format': parser_cnae.get_format_instructions()}
)

prompt_semelhante = PromptTemplate(
    template="""
    Sugira outros códigos CNAE semelhantes ao código: {cnae}.
    {output_format}
    """,
    partial_variables={'output_format': parser_semelhante.get_format_instructions()}
)

prompt_impostos = PromptTemplate(
    template="""Liste quais os impostos que podem incidir para o código CNAE: {cnae}."""
)

# Initialize the OpenAI client
model = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    api_key=os.getenv('OPENAI_API_KEY')
)

# Collect user inputs
atividades = input("Descreva resumidamente as atividades exercidas pela sua empresa: ")

# Define the variables for the prompt
vars = {
    "atividades": atividades,
}

# Create the chains
chain_cnae = prompt_cnae | model | parser_cnae
chain_semelhante = prompt_semelhante | model | parser_semelhante
chain_impostos = prompt_impostos | model | StrOutputParser()

chain_principal = (chain_cnae | chain_semelhante | chain_impostos)

# Invoke the chain with user inputs
response = chain_principal.invoke(
    vars
)

print(response)