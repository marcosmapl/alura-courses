"""
Python script to generate a personalized travel itinerary using OpenAI's API.
"""
# from openai import OpenAI
from langchain_openai import ChatOpenAI
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

prompt = f"Gere um roteiro de viagem de {n_days} dias para {n_adults} adultos e {n_children} crianças em {destination}, com foco em {interests}, durante o período de {period} e com orçamento de R$ {budget}. Inclua atividades diárias, opções de alimentação, hospedagem, dicas práticas e um resumo com ritmo e custo estimado."

client = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    api_key=os.getenv('OPENAI_API_KEY')
)

response = client.invoke(prompt)

print("Roteiro de Viagem:")
print(response.content)