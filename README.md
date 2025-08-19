# Travel Agent

A sophisticated travel planning agent powered by OpenAI and Pydantic AI that supports both single-agent and multi-agent architectures for comprehensive trip planning.

## Installation

### Install from Git

```bash
pip install git+https://github.com/yourusername/travel-agent.git
```

### Install for Development

```bash
git clone https://github.com/yourusername/travel-agent.git
cd travel-agent
pip install -e .
```

## Setup

### Environment Variables

Create a `.env` file in the project root directory with your API keys:

```bash
cp .env.example .env
```

Then edit the `.env` file with your actual API keys:

```bash
# OpenAI API Configuration
OPENAI_API_KEY=your-openai-api-key-here

# Google API Configuration (required for multi-agent mode)
GOOGLE_CLIENT_API_KEY=your-google-client-api-key-here
GOOGLE_CSE_ID=your-google-cse-id-here
GOOGLE_API_KEY=your-google-api-key-here

# Tavily API Configuration (required for multi-agent mode)
TAVILY_API_KEY=your-tavily-api-key-here
```


## Usage

### Command Line Interface

```bash
python run_agent.py --query "what is my current location" --model openai:gpt-4o --env .env
```

### Basic Usage

#### Single-Agent Mode (Traditional)
```python
from agent import TravelAgent
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Create a single-agent instance
agent = TravelAgent(use_multi_agent=False)

# Ask a travel question
response, trace = agent.run("Plan a 3-day trip to Paris")
print(response)
```

#### Multi-Agent Mode (Advanced)
```python
from agent import TravelAgent
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Create a multi-agent instance with specialized agents
agent = TravelAgent(use_multi_agent=True)

# Get agent information
info = agent.get_agent_info()
print(f"Mode: {info['mode']}")
print(f"Available agents: {info['agents']}")

# Ask a comprehensive travel question
response, trace = agent.run("Plan a 5-day trip from Los Angeles to New York including flights, hotels, restaurants, and activities")
print(response)
```

### Web Interface

Launch the interactive web interface with multi-agent support:

```bash
python app.py --port 8050
```

Then open your browser to `http://localhost:8050` and use the multi-agent toggle to switch between modes.

### Example Queries

- "Plan a 5-day trip to Tokyo for a family of 4"
- "What's the best time to visit Greece?"
- "Suggest budget-friendly accommodations in Barcelona"
- "Create an itinerary for a weekend in New York City"

## Features

### Core Features
- AI-powered travel planning with OpenAI and Pydantic AI
- Comprehensive trip recommendations including flights, hotels, restaurants, and activities
- Message tracing for debugging and analysis
- Synchronous API for easy integration
- Interactive web interface with advanced RoPE analysis

### Multi-Agent Architecture
- **Specialized Agents**: Dedicated agents for different travel domains
  - 🛫 **Flight Agent**: Searches for flights and transportation options
  - 🏨 **Hotel Agent**: Finds accommodations and lodging recommendations
  - 🍽️ **Restaurant Agent**: Discovers dining options and local cuisine
  - 🎯 **Activity Agent**: Suggests activities, attractions, and entertainment
  - 🌤️ **Weather Agent**: Provides weather forecasts and travel conditions
  - 🤖 **Coordinator Agent**: Orchestrates all agents and synthesizes responses

### Advanced Analysis
- Enhanced influence chain analysis with RoPE (Rotary Position Embedding)
- Tool execution tracing and visualization
- Agent delegation visualization in multi-agent mode
- Real-time text selection analysis
- O(n²) causal relationship detection

### Web Interface Features
- Toggle between single-agent and multi-agent modes
- Session management and history
- Interactive tool execution browser
- Real-time agent delegation tracking
- Advanced text selection analysis with floating analyzer

## Requirements

- Python 3.8+
- OpenAI API key
- Internet connection

## Dependencies

### Core Dependencies
- `openai>=1.0.0` - OpenAI API client
- `pydantic-ai` - AI agent framework
- `python-dotenv` - Environment variable management

### Multi-Agent Dependencies
- `googlemaps` - Google Maps API integration
- `tavily-python` - Tavily search API client
- `requests` - HTTP requests

### Web Interface Dependencies
- `dash` - Web framework
- `dash-bootstrap-components` - UI components
- `numpy` - Numerical computations
- `scipy` - Scientific computing
- `nest-asyncio` - Async support in Dash

## Testing

Test the multi-agent functionality:

```bash
python test_multi_agent.py
```

This will test:
- Single-agent mode functionality
- Multi-agent mode with coordinator delegation
- Individual specialized agent functionality
- Environment variable validation