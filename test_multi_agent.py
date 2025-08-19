#!/usr/bin/env python3
"""
Test script for multi-agent travel planning functionality.

This script tests both single-agent and multi-agent modes to ensure the implementation works correctly.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the agent directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'agent'))

def test_single_agent():
    """Test the traditional single-agent mode."""
    print("🔧 Testing Single-Agent Mode...")
    try:
        from agent import TravelAgent
        
        # Initialize single-agent mode
        agent = TravelAgent(use_multi_agent=False)
        info = agent.get_agent_info()
        
        print(f"✅ Single-agent initialized successfully")
        print(f"   Mode: {info['mode']}")
        print(f"   Agents: {info['agents']}")
        print(f"   Description: {info['description']}")
        
        # Test with a simple query
        query = "Plan a 2-day trip to San Francisco"
        print(f"\n🧪 Testing query: '{query}'")
        
        response, trace = agent.run(query)
        
        print(f"✅ Query executed successfully")
        print(f"   Response length: {len(response)} characters")
        print(f"   Trace messages: {len(trace)} messages")
        print(f"   Response preview: {response[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Single-agent test failed: {e}")
        return False

def test_multi_agent():
    """Test the new multi-agent mode."""
    print("\n🤖 Testing Multi-Agent Mode...")
    try:
        from agent import TravelAgent
        
        # Initialize multi-agent mode
        agent = TravelAgent(use_multi_agent=True)
        info = agent.get_agent_info()
        
        print(f"✅ Multi-agent initialized successfully")
        print(f"   Mode: {info['mode']}")
        print(f"   Agents: {info['agents']}")
        print(f"   Description: {info['description']}")
        
        # Test with a comprehensive travel query
        query = "Plan a 3-day trip from Los Angeles to New York City, including flights, hotels, restaurants, and activities"
        print(f"\n🧪 Testing query: '{query}'")
        
        response, trace = agent.run(query)
        
        print(f"✅ Query executed successfully")
        print(f"   Response length: {len(response)} characters")
        print(f"   Trace messages: {len(trace)} messages")
        print(f"   Response preview: {response[:200]}...")
        
        # Check if response contains multi-agent indicators
        response_lower = response.lower()
        multi_agent_indicators = [
            'flight agent', 'hotel agent', 'restaurant agent', 
            'activity agent', 'weather agent', 'coordinator'
        ]
        
        found_indicators = [indicator for indicator in multi_agent_indicators if indicator in response_lower]
        if found_indicators:
            print(f"✅ Multi-agent delegation detected: {found_indicators}")
        else:
            print("⚠️  No clear multi-agent delegation indicators found in response")
        
        return True
        
    except Exception as e:
        print(f"❌ Multi-agent test failed: {e}")
        return False

def test_specialized_agents():
    """Test individual specialized agents."""
    print("\n🎯 Testing Specialized Agents...")
    try:
        from agent.multi_agent import (
            create_travel_dependencies, flight_agent, hotel_agent, 
            restaurant_agent, activity_agent, weather_agent
        )
        
        # Create dependencies
        deps = create_travel_dependencies()
        print("✅ Travel dependencies created")
        
        # Test flight agent
        print("\n✈️ Testing flight agent...")
        flight_result = flight_agent.run_sync(
            "Find flights from Los Angeles to New York on January 15th",
            deps=deps
        )
        print(f"   Flight agent response: {str(flight_result.output)[:100]}...")
        
        # Test hotel agent
        print("\n🏨 Testing hotel agent...")
        hotel_result = hotel_agent.run_sync(
            "Find hotels in New York City from January 15 to January 17 for 2 guests",
            deps=deps
        )
        print(f"   Hotel agent response: {str(hotel_result.output)[:100]}...")
        
        # Test restaurant agent
        print("\n🍽️ Testing restaurant agent...")
        restaurant_result = restaurant_agent.run_sync(
            "Find good restaurants in New York City",
            deps=deps
        )
        print(f"   Restaurant agent response: {str(restaurant_result.output)[:100]}...")
        
        # Test activity agent  
        print("\n🎯 Testing activity agent...")
        activity_result = activity_agent.run_sync(
            "Find activities and attractions in New York City",
            deps=deps
        )
        print(f"   Activity agent response: {str(activity_result.output)[:100]}...")
        
        # Test weather agent
        print("\n🌤️ Testing weather agent...")
        weather_result = weather_agent.run_sync(
            "Get weather forecast for New York City for 7 days",
            deps=deps
        )
        print(f"   Weather agent response: {str(weather_result.output)[:100]}...")
        
        print("✅ All specialized agents tested successfully")
        return True
        
    except Exception as e:
        print(f"❌ Specialized agents test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Multi-Agent Travel Planning System Tests")
    print("=" * 50)
    
    # Check environment variables
    required_vars = ["GOOGLE_CLIENT_API_KEY", "TAVILY_API_KEY", "OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing environment variables: {missing_vars}")
        print("Please set these in your .env file")
        return False
    
    print("✅ All required environment variables found")
    
    # Run tests
    tests = [
        test_single_agent,
        test_multi_agent,
        test_specialized_agents
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Multi-agent system is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)