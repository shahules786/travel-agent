from pathlib import Path
from pydantic_ai import Agent
from typing import Optional

MODEL = 'openai:gpt-4o'

prompt_file = Path(__file__).parent / "prompt.txt"
with open(prompt_file, "r", encoding="utf-8") as file:
    SYSTEM_PROMPT = file.read()

agent = Agent(system_prompt=SYSTEM_PROMPT, instrument=True)

_TOOLS_REGISTERED = False

class TravelAgent:
    """
    Travel Agent supporting both single-agent and multi-agent modes.
    
    The single-agent mode uses the original implementation with all tools registered to one agent.
    The multi-agent mode uses specialized agents that delegate to each other for different travel tasks.
    """
    
    def __init__(self, use_multi_agent: bool = False):
        global _TOOLS_REGISTERED
        self.use_multi_agent = use_multi_agent
        
        if use_multi_agent:
            # Initialize multi-agent system
            try:
                from .multi_agent import create_travel_dependencies
                from .coordinator import MultiAgentTravelAgent
                
                self.dependencies = create_travel_dependencies()
                self.multi_agent = MultiAgentTravelAgent(self.dependencies)
                self.agent = None  # Multi-agent doesn't use single agent
                
            except Exception as e:
                print(f"Warning: Could not initialize multi-agent system: {e}")
                print("Falling back to single-agent mode...")
                self.use_multi_agent = False
                self._init_single_agent()
        else:
            self._init_single_agent()
    
    def _init_single_agent(self):
        """Initialize single-agent mode with traditional tools."""
        global _TOOLS_REGISTERED
        self.agent = agent
        
        if not _TOOLS_REGISTERED:
            from .tools import register_tools
            register_tools(self.agent)
            _TOOLS_REGISTERED = True

    def run(self, query: str, model: str = MODEL):
        """
        Run the travel agent in either single-agent or multi-agent mode.
        
        Args:
            query: User's travel query
            model: Model to use
            
        Returns:
            Tuple of (response_text, messages_json)
        """
        if self.use_multi_agent and hasattr(self, 'multi_agent'):
            return self.multi_agent.run(query, model)
        else:
            # Traditional single-agent mode
            response = self.agent.run_sync(query, model=model)
            return response.output, response.all_messages_json()
    
    def get_agent_info(self) -> dict:
        """Get information about the current agent configuration."""
        if self.use_multi_agent:
            return {
                "mode": "multi-agent",
                "agents": ["coordinator", "flight", "hotel", "restaurant", "activity", "weather"],
                "description": "Uses specialized agents that delegate tasks to each other"
            }
        else:
            return {
                "mode": "single-agent", 
                "agents": ["travel_agent"],
                "description": "Uses single agent with all tools registered"
            }