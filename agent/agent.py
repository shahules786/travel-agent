from pathlib import Path
from pydantic_ai import Agent

MODEL = 'openai:gpt-4o'

prompt_file = Path(__file__).parent / "prompt.txt"
with open(prompt_file, "r") as file:
    SYSTEM_PROMPT = file.read()

agent = Agent(system_prompt=SYSTEM_PROMPT, instrument=True)



class TravelAgent:
    def __init__(self):
        self.agent = agent

    def run(self, query: str, model: str = MODEL):
        
        response = self.agent.run_sync(query, model=model)

        return response.output, response.all_messages_json()