from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import os

from pydantic_ai import Agent, RunContext
from pydantic import BaseModel, Field
import googlemaps
import requests
from datetime import datetime
from tavily import TavilyClient

# Shared dependencies
@dataclass
class TravelDependencies:
    gmaps: googlemaps.Client
    tavily_client: TavilyClient

# Output models for structured responses
class FlightResult(BaseModel):
    """Flight search results"""
    origin: str
    destination: str
    departure_date: str
    flights: List[Dict[str, Any]]
    total_duration: str
    estimated_cost: str

class HotelResult(BaseModel):
    """Hotel search results"""
    location: str
    check_in: str
    check_out: str
    hotels: List[Dict[str, Any]]
    price_range: str

class RestaurantResult(BaseModel):
    """Restaurant recommendations"""
    location: str
    restaurants: List[Dict[str, Any]]
    cuisine_types: List[str]

class ActivityResult(BaseModel):
    """Activity and attraction recommendations"""
    location: str
    activities: List[Dict[str, Any]]
    categories: List[str]

class Failed(BaseModel):
    """Unable to find satisfactory results"""
    reason: str

# ==================== SPECIALIZED AGENTS ====================

# Flight Search Agent
flight_agent = Agent[TravelDependencies, Union[FlightResult, Failed]](
    'openai:gpt-4o',
    deps_type=TravelDependencies,
    output_type=Union[FlightResult, Failed],
    system_prompt="""You are a specialized flight search agent. Your role is to find and recommend flights between locations.

Use the flight_search tool to find flights and provide comprehensive flight options with:
- Multiple airline options
- Different departure times
- Price comparisons
- Duration estimates
- Connection information

Always provide at least 3 flight options when available, covering budget, mid-range, and premium choices.
"""
)

@flight_agent.tool
def flight_search(ctx: RunContext[TravelDependencies], origin: str, destination: str, 
                 departure_date: str, return_date: Optional[str] = None) -> Dict[str, Any]:
    """
    Search for flights between origin and destination.
    
    Args:
        origin: Departure city or airport code
        destination: Arrival city or airport code  
        departure_date: Departure date (YYYY-MM-DD)
        return_date: Optional return date for round-trip
    """
    # Use web search to find flight information
    search_query = f"flights from {origin} to {destination} {departure_date}"
    if return_date:
        search_query += f" return {return_date}"
    
    search_results = ctx.deps.tavily_client.search(search_query, max_results=10)
    
    # Also get directions for reference
    try:
        directions = ctx.deps.gmaps.directions(origin, destination, mode="driving", departure_time=datetime.now())
        driving_info = {
            "driving_distance": directions[0]['legs'][0]['distance']['text'] if directions else "Unknown",
            "driving_duration": directions[0]['legs'][0]['duration']['text'] if directions else "Unknown"
        }
    except:
        driving_info = {"driving_distance": "Unknown", "driving_duration": "Unknown"}
    
    return {
        "search_results": search_results,
        "driving_alternative": driving_info,
        "origin": origin,
        "destination": destination,
        "departure_date": departure_date,
        "return_date": return_date
    }

# Hotel Search Agent
hotel_agent = Agent[TravelDependencies, Union[HotelResult, Failed]](
    'openai:gpt-4o',
    deps_type=TravelDependencies,
    output_type=Union[HotelResult, Failed],
    system_prompt="""You are a specialized hotel and accommodation search agent. Your role is to find and recommend accommodations.

Use the hotel_search tool to find hotels and provide comprehensive accommodation options with:
- Multiple hotel categories (budget, mid-range, luxury)
- Detailed amenities and features
- Location advantages (proximity to attractions, transportation)
- Guest ratings and reviews
- Booking information and pricing

Always provide at least 3 accommodation options covering different budget ranges.
"""
)

@hotel_agent.tool
def hotel_search(ctx: RunContext[TravelDependencies], location: str, check_in: str, 
                check_out: str, guests: int = 2) -> Dict[str, Any]:
    """
    Search for hotels and accommodations in a location.
    
    Args:
        location: City or area to search for hotels
        check_in: Check-in date (YYYY-MM-DD)
        check_out: Check-out date (YYYY-MM-DD)
        guests: Number of guests
    """
    # Geocode the location first
    geocode_results = ctx.deps.gmaps.geocode(location)
    if geocode_results:
        lat = geocode_results[0]['geometry']['location']['lat']
        lng = geocode_results[0]['geometry']['location']['lng']
        
        # Search for hotels using Google Places
        hotels = ctx.deps.gmaps.places(
            query=f"hotels in {location}",
            location=(lat, lng),
            radius=5000
        ).get('results', [])
    else:
        hotels = []
    
    # Web search for additional hotel information
    search_query = f"best hotels {location} {check_in} {check_out} booking"
    web_results = ctx.deps.tavily_client.search(search_query, max_results=8)
    
    return {
        "location": location,
        "check_in": check_in,
        "check_out": check_out,
        "guests": guests,
        "google_places_hotels": hotels,
        "web_search_results": web_results
    }

# Restaurant Search Agent  
restaurant_agent = Agent[TravelDependencies, Union[RestaurantResult, Failed]](
    'openai:gpt-4o',
    deps_type=TravelDependencies,
    output_type=Union[RestaurantResult, Failed],
    system_prompt="""You are a specialized restaurant and dining recommendation agent. Your role is to find and recommend restaurants and dining experiences.

Use the restaurant_search tool to find restaurants and provide comprehensive dining options with:
- Multiple cuisine types and dining styles
- Restaurant details (hours, price range, specialties)
- Location and accessibility information
- User ratings and popular dishes
- Reservation information

Always provide diverse dining options covering different price ranges and cuisine types.
"""
)

@restaurant_agent.tool
def restaurant_search(ctx: RunContext[TravelDependencies], location: str, 
                     cuisine_type: Optional[str] = None, price_range: Optional[str] = None) -> Dict[str, Any]:
    """
    Search for restaurants in a location.
    
    Args:
        location: City or specific area to search
        cuisine_type: Optional cuisine type filter
        price_range: Optional price range (budget, mid-range, fine-dining)
    """
    # Geocode the location
    geocode_results = ctx.deps.gmaps.geocode(location)
    if geocode_results:
        lat = geocode_results[0]['geometry']['location']['lat']
        lng = geocode_results[0]['geometry']['location']['lng']
        
        # Search for restaurants using Google Places
        query = "restaurants"
        if cuisine_type:
            query = f"{cuisine_type} restaurants"
        
        restaurants = ctx.deps.gmaps.places(
            query=f"{query} in {location}",
            location=(lat, lng),
            radius=5000
        ).get('results', [])
    else:
        restaurants = []
    
    # Web search for restaurant recommendations
    search_query = f"best restaurants {location}"
    if cuisine_type:
        search_query += f" {cuisine_type}"
    if price_range:
        search_query += f" {price_range}"
    
    web_results = ctx.deps.tavily_client.search(search_query, max_results=8)
    
    return {
        "location": location,
        "cuisine_type": cuisine_type,
        "price_range": price_range,
        "google_places_restaurants": restaurants,
        "web_search_results": web_results
    }

# Activity Search Agent
activity_agent = Agent[TravelDependencies, Union[ActivityResult, Failed]](
    'openai:gpt-4o',
    deps_type=TravelDependencies,
    output_type=Union[ActivityResult, Failed],
    system_prompt="""You are a specialized activity and attraction recommendation agent. Your role is to find and recommend activities, attractions, and experiences.

Use the activity_search tool to find activities and provide comprehensive activity options with:
- Tourist attractions and landmarks
- Cultural experiences and museums
- Outdoor activities and parks
- Entertainment and events
- Activity details (hours, pricing, duration, booking requirements)

Always provide diverse activity options covering different interests and activity levels.
"""
)

@activity_agent.tool
def activity_search(ctx: RunContext[TravelDependencies], location: str, 
                   activity_type: Optional[str] = None, duration: Optional[str] = None) -> Dict[str, Any]:
    """
    Search for activities and attractions in a location.
    
    Args:
        location: City or area to search for activities
        activity_type: Optional activity type (museum, outdoor, entertainment, etc.)
        duration: Optional duration preference (half-day, full-day, etc.)
    """
    # Geocode the location
    geocode_results = ctx.deps.gmaps.geocode(location)
    if geocode_results:
        lat = geocode_results[0]['geometry']['location']['lat']
        lng = geocode_results[0]['geometry']['location']['lng']
        
        # Search for attractions using Google Places
        query = "attractions"
        if activity_type:
            query = f"{activity_type} attractions"
        
        attractions = ctx.deps.gmaps.places(
            query=f"{query} in {location}",
            location=(lat, lng),
            radius=8000
        ).get('results', [])
        
        # Also search for specific points of interest
        tourist_spots = ctx.deps.gmaps.places(
            query=f"tourist attractions {location}",
            location=(lat, lng),
            radius=8000
        ).get('results', [])
    else:
        attractions = []
        tourist_spots = []
    
    # Web search for activity recommendations
    search_query = f"things to do {location} attractions activities"
    if activity_type:
        search_query += f" {activity_type}"
    
    web_results = ctx.deps.tavily_client.search(search_query, max_results=10)
    
    return {
        "location": location,
        "activity_type": activity_type,
        "duration": duration,
        "google_places_attractions": attractions,
        "google_places_tourist_spots": tourist_spots,
        "web_search_results": web_results
    }

# Weather Agent
weather_agent = Agent[TravelDependencies, Dict[str, Any]](
    'openai:gpt-4o',
    deps_type=TravelDependencies,
    system_prompt="""You are a specialized weather information agent. Your role is to provide comprehensive weather forecasts and travel-related weather advice.

Use the weather_forecast tool to get weather information and provide:
- Current weather conditions
- Multi-day forecasts
- Weather-based travel recommendations
- Packing suggestions based on weather
- Activity recommendations based on weather conditions
"""
)

@weather_agent.tool
def weather_forecast(ctx: RunContext[TravelDependencies], location: str, days: int = 7) -> Dict[str, Any]:
    """
    Get weather forecast for a location.
    
    Args:
        location: City or location for weather forecast
        days: Number of days to forecast
    """
    # Geocode location to get coordinates
    geocode_results = ctx.deps.gmaps.geocode(location)
    if not geocode_results:
        return {"error": f"Could not find location: {location}"}
    
    lat = geocode_results[0]['geometry']['location']['lat']
    lng = geocode_results[0]['geometry']['location']['lng']
    
    # Try to get weather using Google API (may need valid API key)
    try:
        params = {
            "key": os.environ.get("GOOGLE_API_KEY", ""),
            "location.latitude": lat,
            "location.longitude": lng,
        }
        
        # Current weather
        current_url = "https://weather.googleapis.com/v1/currentConditions:lookup"
        current_response = requests.get(current_url, params=params)
        current_weather = current_response.json() if current_response.status_code == 200 else {}
        
        # Forecast
        forecast_params = params.copy()
        forecast_params["days"] = days
        forecast_url = "https://weather.googleapis.com/v1/forecast/days:lookup"
        forecast_response = requests.get(forecast_url, params=forecast_params)
        forecast_weather = forecast_response.json() if forecast_response.status_code == 200 else {}
        
    except Exception as e:
        current_weather = {"error": f"Weather API error: {str(e)}"}
        forecast_weather = {"error": f"Weather API error: {str(e)}"}
    
    # Also use web search as backup
    weather_search = ctx.deps.tavily_client.search(f"weather forecast {location} {days} days", max_results=5)
    
    return {
        "location": location,
        "coordinates": {"lat": lat, "lng": lng},
        "current_weather": current_weather,
        "forecast": forecast_weather,
        "web_search_weather": weather_search,
        "days": days
    }

# ==================== INITIALIZATION FUNCTION ====================

def create_travel_dependencies() -> TravelDependencies:
    """Create shared dependencies for all travel agents."""
    gmaps = googlemaps.Client(key=os.environ["GOOGLE_CLIENT_API_KEY"])
    tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    
    return TravelDependencies(
        gmaps=gmaps,
        tavily_client=tavily_client
    )