from pathlib import Path
from pydantic_ai import Agent
from agent.tracer import covert_to_trace

MODEL = 'openai:gpt-4o'

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

prompt_file = Path(__file__).parent / "prompt.txt"
with open(prompt_file, "r") as file:
    SYSTEM_PROMPT = file.read()

agent = Agent(model=MODEL, system_prompt=SYSTEM_PROMPT, instrument=True)



class TravelAgent:
    def __init__(self):
        self.agent = agent

    def run(self, query: str, model: str = MODEL):
        
        response = self.agent.run_sync(query, model=model)
        return response.output, covert_to_trace(response.all_messages())