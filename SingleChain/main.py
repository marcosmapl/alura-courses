"""
Python script to generate a personalized travel itinerary using OpenAI's API.
"""
# from openai import OpenAI
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.globals import set_debug
from pydantic import BaseModel, Field

import os


class Destino(BaseModel):
    cidade: str = Field("A cidade recomendada para visitar.")
    motivo: str = Field("O motivo pelo qual a cidade foi recomendada.")


class Restaurantes(BaseModel):
    cidade: str = Field("A cidade recomendada para visitar.")
    restaurantes: list[str] = Field("Uma lista com os nomes dos restaurantes recomendados na cidade.")


set_debug(True)
load_dotenv()

# Initialize the JSON output parser
parser_destino = JsonOutputParser(pydantic_object=Destino)
parser_restaurante = JsonOutputParser(pydantic_object=Restaurantes)

# Define the prompt template
prompt_destino = PromptTemplate(
    template="""
    Sugira uma cidade para visitar dado o meu interesse por {interests}.
    {output_format}
    """,
    input_variables=["interests"],
    partial_variables={'output_format': parser_destino.get_format_instructions()}
)

prompt_restaurante = PromptTemplate(
    template="""
    Sugira restaurantes populares da cidade {cidade}.
    {output_format}
    """,
    partial_variables={'output_format': parser_restaurante.get_format_instructions()}
)

prompt_cultural = PromptTemplate(
    template="""Sugira atividades culturais para fazer em {cidade}."""
)

# Initialize the OpenAI client
model = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    api_key=os.getenv('OPENAI_API_KEY')
)

# Collect user inputs
interests = input("Quais são os interesses da sua viagem? (ex: aventura, cultura, gastronomia): ")

# Define the variables for the prompt
vars = {
    "interests": interests,
}

# Create the chains
chain_destino = prompt_destino | model | parser_destino
chain_restaurante = prompt_restaurante | model | parser_restaurante
chain_cultural = prompt_cultural | model | StrOutputParser()

chain_viajai = (chain_destino | chain_restaurante | chain_cultural)

# Invoke the chain with user inputs
response = chain_viajai.invoke(
    vars
)

print(response)