"""
Python script to generate a personalized travel itinerary using OpenAI's API.
"""
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

n_days = input("Digite o número de dias da viagem: ")
n_adults = input("Digite o número de adultos na família: ")
n_children = input("Digite o número de crianças na família: ")
destination = input("Digite o destino da viagem: ")
interests = input("Quais são os interesses da sua família? (ex: aventura, cultura, gastronomia): ")

prompt = f"Crie um roteiro de viagens de {n_days} dias para {destination}. Minha família gosta de {interests}. Somos {n_adults} adultos e {n_children} crianças."

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "system",
            "content": "Você é um assistente que ajuda a criar roteiros de viagem personalizados.",
        },
        {
            "role": "user",
            "content": prompt,
        }
    ])

print("Roteiro de Viagem:")
print(response)
print(response.choices)
print(response.choices[0].message.content)