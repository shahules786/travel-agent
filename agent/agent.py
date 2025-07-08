
import os
from pathlib import Path

from openai import AsyncOpenAI

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from agent.tracer import covert_to_trace

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

client = AsyncOpenAI(max_retries=3)
model = OpenAIModel('gpt-4o', provider=OpenAIProvider(openai_client=client))

prompt_file = Path(__file__).parent / "prompt.txt"
with open(prompt_file, "r") as file:
    SYSTEM_PROMPT = file.read()

travel_agent = Agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
)


class TravelAgent:
    def __init__(self):
        self.agent = travel_agent

    def run(self, query: str):
        response = self.agent.run_sync(query)
        return response.output, covert_to_trace(response.all_messages())