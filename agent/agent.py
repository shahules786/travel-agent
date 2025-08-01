from pathlib import Path
from pydantic_ai import Agent

MODEL = 'openai:gpt-4o'

prompt_file = Path(__file__).parent / "prompt.txt"
with open(prompt_file, "r", encoding="utf-8") as file:
    SYSTEM_PROMPT = file.read()

agent = Agent(system_prompt=SYSTEM_PROMPT, instrument=True)

_TOOLS_REGISTERED = False

class TravelAgent:
    def __init__(self):
        global _TOOLS_REGISTERED
        self.agent = agent
        
        if not _TOOLS_REGISTERED:
            from .tools import register_tools
            register_tools(self.agent)
            _TOOLS_REGISTERED = True

    def run(self, query: str, model: str = MODEL):
        response = self.agent.run_sync(query, model=model)
        return response.output, response.all_messages_json()