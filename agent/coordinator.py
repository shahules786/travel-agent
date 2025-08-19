from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import json

from pydantic_ai import Agent, RunContext
from pydantic import BaseModel

from .multi_agent import (
    TravelDependencies, 
    flight_agent, hotel_agent, restaurant_agent, activity_agent, weather_agent,
    FlightResult, HotelResult, RestaurantResult, ActivityResult, Failed
)

# Coordinator response model
class TravelPlan(BaseModel):
    """Complete travel plan with all components"""
    query: str
    destination: str
    duration: str
    flight_recommendations: Optional[Dict[str, Any]] = None
    hotel_recommendations: Optional[Dict[str, Any]] = None
    restaurant_recommendations: Optional[Dict[str, Any]] = None
    activity_recommendations: Optional[Dict[str, Any]] = None
    weather_forecast: Optional[Dict[str, Any]] = None
    daily_itinerary: List[Dict[str, Any]] = []
    total_estimated_cost: Optional[str] = None

# Main coordinator agent
travel_coordinator = Agent[TravelDependencies, TravelPlan](
    'openai:gpt-4o',
    deps_type=TravelDependencies,
    output_type=TravelPlan,
    system_prompt="""You are the Travel Coordinator Agent, the central orchestrator for comprehensive travel planning.

Your role is to:
1. Analyze the user's travel query to understand their needs
2. Delegate specific tasks to specialized agents using the available tools
3. Synthesize all agent responses into a comprehensive travel plan
4. Create a detailed day-by-day itinerary

Available specialized agents via tools:
- flight_search_delegate: Find flights and transportation options
- hotel_search_delegate: Find accommodations and lodging
- restaurant_search_delegate: Find dining recommendations
- activity_search_delegate: Find activities and attractions
- weather_forecast_delegate: Get weather information

Process:
1. Parse the user query to extract: origin, destination, dates, duration, preferences
2. Call the weather agent first to understand conditions
3. Delegate to flight, hotel, restaurant, and activity agents based on the query
4. Synthesize all responses into a coherent travel plan
5. Create a structured daily itinerary with specific times and locations

Always provide a comprehensive response that includes practical details like:
- Transportation options and costs
- Accommodation recommendations in different price ranges
- Daily activity schedules with specific times
- Restaurant recommendations for each meal
- Weather-appropriate suggestions
- Backup plans for bad weather

Make the plan actionable and detailed enough that travelers can follow it step-by-step.
"""
)

@travel_coordinator.tool
async def flight_search_delegate(ctx: RunContext[TravelDependencies], origin: str, destination: str, 
                                departure_date: str, return_date: Optional[str] = None) -> Dict[str, Any]:
    """
    Delegate flight search to the specialized flight agent.
    
    Args:
        origin: Departure location
        destination: Arrival location
        departure_date: Departure date
        return_date: Optional return date
    """
    result = await flight_agent.run(
        f"Find flights from {origin} to {destination} departing {departure_date}" + 
        (f" returning {return_date}" if return_date else ""),
        deps=ctx.deps,
        usage=ctx.usage
    )
    
    return {
        "agent": "flight_agent",
        "query": f"Flights {origin} → {destination}",
        "result": result.output.model_dump() if hasattr(result.output, 'model_dump') else str(result.output),
        "success": not isinstance(result.output, Failed)
    }

@travel_coordinator.tool
async def hotel_search_delegate(ctx: RunContext[TravelDependencies], location: str, check_in: str, 
                               check_out: str, guests: int = 2) -> Dict[str, Any]:
    """
    Delegate hotel search to the specialized hotel agent.
    
    Args:
        location: Destination city or area
        check_in: Check-in date
        check_out: Check-out date
        guests: Number of guests
    """
    result = await hotel_agent.run(
        f"Find hotels in {location} from {check_in} to {check_out} for {guests} guests",
        deps=ctx.deps,
        usage=ctx.usage
    )
    
    return {
        "agent": "hotel_agent",
        "query": f"Hotels in {location}",
        "result": result.output.model_dump() if hasattr(result.output, 'model_dump') else str(result.output),
        "success": not isinstance(result.output, Failed)
    }

@travel_coordinator.tool
async def restaurant_search_delegate(ctx: RunContext[TravelDependencies], location: str, 
                                   cuisine_type: Optional[str] = None, price_range: Optional[str] = None) -> Dict[str, Any]:
    """
    Delegate restaurant search to the specialized restaurant agent.
    
    Args:
        location: Destination city or area
        cuisine_type: Type of cuisine (optional)
        price_range: Price range preference (optional)
    """
    query = f"Find restaurants in {location}"
    if cuisine_type:
        query += f" serving {cuisine_type} cuisine"
    if price_range:
        query += f" in {price_range} price range"
    
    result = await restaurant_agent.run(
        query,
        deps=ctx.deps,
        usage=ctx.usage
    )
    
    return {
        "agent": "restaurant_agent", 
        "query": f"Restaurants in {location}",
        "result": result.output.model_dump() if hasattr(result.output, 'model_dump') else str(result.output),
        "success": not isinstance(result.output, Failed)
    }

@travel_coordinator.tool
async def activity_search_delegate(ctx: RunContext[TravelDependencies], location: str, 
                                 activity_type: Optional[str] = None, duration: Optional[str] = None) -> Dict[str, Any]:
    """
    Delegate activity search to the specialized activity agent.
    
    Args:
        location: Destination city or area
        activity_type: Type of activities (optional)
        duration: Duration preference (optional)
    """
    query = f"Find activities and attractions in {location}"
    if activity_type:
        query += f" focusing on {activity_type}"
    if duration:
        query += f" suitable for {duration}"
    
    result = await activity_agent.run(
        query,
        deps=ctx.deps,
        usage=ctx.usage
    )
    
    return {
        "agent": "activity_agent",
        "query": f"Activities in {location}",
        "result": result.output.model_dump() if hasattr(result.output, 'model_dump') else str(result.output),
        "success": not isinstance(result.output, Failed)
    }

@travel_coordinator.tool
async def weather_forecast_delegate(ctx: RunContext[TravelDependencies], location: str, days: int = 7) -> Dict[str, Any]:
    """
    Delegate weather forecast to the specialized weather agent.
    
    Args:
        location: Destination city
        days: Number of days to forecast
    """
    result = await weather_agent.run(
        f"Get weather forecast for {location} for {days} days",
        deps=ctx.deps,
        usage=ctx.usage
    )
    
    return {
        "agent": "weather_agent",
        "query": f"Weather for {location}",
        "result": result.output,
        "success": True
    }

# Enhanced TravelAgent class that uses the multi-agent coordinator
class MultiAgentTravelAgent:
    """Enhanced travel agent using multi-agent delegation pattern."""
    
    def __init__(self, dependencies: TravelDependencies):
        self.coordinator = travel_coordinator
        self.deps = dependencies
    
    def run(self, query: str, model: str = 'openai:gpt-4o'):
        """
        Run the multi-agent travel planning system.
        
        Args:
            query: User's travel query
            model: Model to use for the coordinator
            
        Returns:
            Tuple of (response_text, messages_json)
        """
        try:
            # Run the coordinator agent
            result = self.coordinator.run_sync(
                query, 
                deps=self.deps,
                model=model
            )
            
            # Format the response
            if hasattr(result.output, 'model_dump'):
                output_dict = result.output.model_dump()
                response_text = self._format_travel_plan(output_dict)
            else:
                response_text = str(result.output)
            
            # Return formatted response and messages
            return response_text, result.all_messages_json()
            
        except Exception as e:
            error_response = f"Error in multi-agent travel planning: {str(e)}"
            return error_response, []
    
    def _format_travel_plan(self, plan: Dict[str, Any]) -> str:
        """Format the travel plan into a readable markdown response."""
        
        md_response = f"""# 🌍 Multi-Agent Travel Plan

## 📋 Trip Overview
- **Query**: {plan.get('query', 'N/A')}
- **Destination**: {plan.get('destination', 'N/A')}
- **Duration**: {plan.get('duration', 'N/A')}

---

"""
        
        # Add flight recommendations
        if plan.get('flight_recommendations'):
            md_response += "## ✈️ Flight Recommendations\n"
            flight_data = plan['flight_recommendations']
            if isinstance(flight_data, dict) and flight_data.get('success'):
                md_response += f"*Provided by Flight Agent*\n\n{self._format_section(flight_data.get('result', {}))}\n\n"
            else:
                md_response += "*No flight recommendations available*\n\n"
        
        # Add hotel recommendations
        if plan.get('hotel_recommendations'):
            md_response += "## 🏨 Hotel Recommendations\n"
            hotel_data = plan['hotel_recommendations']
            if isinstance(hotel_data, dict) and hotel_data.get('success'):
                md_response += f"*Provided by Hotel Agent*\n\n{self._format_section(hotel_data.get('result', {}))}\n\n"
            else:
                md_response += "*No hotel recommendations available*\n\n"
        
        # Add restaurant recommendations
        if plan.get('restaurant_recommendations'):
            md_response += "## 🍽️ Restaurant Recommendations\n"
            restaurant_data = plan['restaurant_recommendations']
            if isinstance(restaurant_data, dict) and restaurant_data.get('success'):
                md_response += f"*Provided by Restaurant Agent*\n\n{self._format_section(restaurant_data.get('result', {}))}\n\n"
            else:
                md_response += "*No restaurant recommendations available*\n\n"
        
        # Add activity recommendations
        if plan.get('activity_recommendations'):
            md_response += "## 🎯 Activity Recommendations\n"
            activity_data = plan['activity_recommendations']
            if isinstance(activity_data, dict) and activity_data.get('success'):
                md_response += f"*Provided by Activity Agent*\n\n{self._format_section(activity_data.get('result', {}))}\n\n"
            else:
                md_response += "*No activity recommendations available*\n\n"
        
        # Add weather forecast
        if plan.get('weather_forecast'):
            md_response += "## 🌤️ Weather Forecast\n"
            weather_data = plan['weather_forecast']
            if isinstance(weather_data, dict) and weather_data.get('success'):
                md_response += f"*Provided by Weather Agent*\n\n{self._format_section(weather_data.get('result', {}))}\n\n"
            else:
                md_response += "*No weather information available*\n\n"
        
        # Add daily itinerary
        if plan.get('daily_itinerary'):
            md_response += "## 📅 Daily Itinerary\n"
            for i, day in enumerate(plan['daily_itinerary'], 1):
                md_response += f"### Day {i}\n{self._format_section(day)}\n\n"
        
        # Add cost estimate
        if plan.get('total_estimated_cost'):
            md_response += f"## 💰 Estimated Total Cost\n{plan['total_estimated_cost']}\n\n"
        
        md_response += "---\n*Generated by Multi-Agent Travel Planning System*\n"
        md_response += "*🤖 Agents: Flight • Hotel • Restaurant • Activity • Weather • Coordinator*"
        
        return md_response
    
    def _format_section(self, data: Any) -> str:
        """Format a section of data into readable text."""
        if isinstance(data, dict):
            if 'error' in data:
                return f"⚠️ {data['error']}"
            
            formatted = ""
            for key, value in data.items():
                if key in ['google_places_hotels', 'google_places_restaurants', 'google_places_attractions']:
                    if value and len(value) > 0:
                        formatted += f"**{key.replace('_', ' ').title()}:**\n"
                        for item in value[:3]:  # Show top 3
                            name = item.get('name', 'Unknown')
                            rating = item.get('rating', 'N/A')
                            formatted += f"- {name} (Rating: {rating})\n"
                        formatted += "\n"
                elif key == 'web_search_results' and isinstance(value, dict):
                    results = value.get('results', [])
                    if results:
                        formatted += "**Web Search Results:**\n"
                        for result in results[:3]:  # Show top 3
                            title = result.get('title', 'No title')
                            url = result.get('url', '#')
                            formatted += f"- [{title}]({url})\n"
                        formatted += "\n"
                elif isinstance(value, (str, int, float)) and not key.startswith('_'):
                    formatted += f"**{key.replace('_', ' ').title()}:** {value}\n"
            
            return formatted if formatted else str(data)
        
        elif isinstance(data, list):
            return "\n".join([f"- {item}" for item in data[:5]])  # Show top 5
        
        else:
            return str(data)