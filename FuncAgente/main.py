from dotenv import load_dotenv

from getpass import getpass

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import tool, ToolRuntime
from langchain_core.prompts import load_prompt
from langchain_openai import ChatOpenAI

from langgraph.checkpoint.memory import InMemorySaver

from pydantic import BaseModel

import os


class Context(BaseModel):
    """Custom runtime context schema."""
    # User identifier
    user_id: str


class ResponseFormat(BaseModel):
    """Response schema for the agent."""
    # A punny response (always required)
    punny_response: str
    # Any interesting information about the weather if available
    weather_conditions: str | None = None

@tool
def get_weather_for_location(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """Retrieve user information based on user ID."""
    user_id = runtime.context.user_id
    return "Florida" if user_id == "1" else "SF"

if __name__ == "__main__":
    load_dotenv()

    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass("Enter your OpenAI API key: ")


    llm_api_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("OPENAI_MODEL") or ""
    print(f"Using model: {model_name}")
    print(f"LLM API Key: {llm_api_key}")


    model = ChatOpenAI(model=model_name, openai_api_key=llm_api_key, temperature=0.7)
    checkpointer = InMemorySaver()

    system_prompt = load_prompt(r"prompts\weater_system_template.json").format()

    agent = create_agent(
        model=model,
        system_prompt=system_prompt,
        tools=[get_weather_for_location, get_user_location],
        context_schema=Context,
        response_format=ToolStrategy(ResponseFormat),
        checkpointer=checkpointer,
    )

    # `thread_id` is a unique identifier for a given conversation.
    config = {"configurable": {"thread_id": "1"}}

    response = agent.invoke(
        {"messages": [{"role": "user", "content": "what is the weather outside?"}]},
        config=config,
        context=Context(user_id="1")
    )

    print(response['structured_response'])

    response = agent.invoke(
        {"messages": [{"role": "user", "content": "thank you!"}]},
        config=config,
        context=Context(user_id="1")
    )

    print(response['structured_response'])