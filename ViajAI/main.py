"""
Python script to generate a personalized travel itinerary using OpenAI's API.
"""
# from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

n_days = input("Digite o número de dias da viagem: ")
n_adults = input("Digite o número de adultos na família: ")
n_children = input("Digite o número de crianças na família: ")
destination = input("Digite o destino da viagem: ")
interests = input("Quais são os interesses da sua família? (ex: aventura, cultura, gastronomia): ")
period = input("Digite o(s) mês(es) da viagem: ")
budget = input("Digite o orçamento aproximado da viagem (em R$): ")

template = PromptTemplate(
    tmplate="""
    Crie um roteiro de viagem para uma família de {n_adults} adultos e {n_children} crianças.
    A viagem terá como destino {destination}, duração de {n_days} dias nos meses {period}.
    A família tem interesse em {interests} e pretende gastar até R$ {budget} na viagem.
    Inclua atividades diárias, opções de alimentação, hospedagem, dicas práticas e um resumo com ritmo e custo estimado.
    """
)

prompt = template.format(
    n_adults=n_adults,
    n_children=n_children,
    destination=destination,
    n_days=n_days,
    interests=interests,
    period=period,
    budget=budget
)

print("Prompt gerado:", prompt)

# client = ChatOpenAI(
#     model="gpt-3.5-turbo",
#     temperature=0.7,
#     api_key=os.getenv('OPENAI_API_KEY')
# )

# response = client.invoke(prompt)

# print("Roteiro de Viagem:")
# print(response.content)