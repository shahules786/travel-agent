#!/usr/bin/env python3
"""
Test script for complex travel queries using the enhanced travel agent.
This script tests various scenarios to ensure the 20-30 step planning works correctly.
"""

import os
import sys
from agent import TravelAgent
from load_dotenv import load_dotenv
import json
from datetime import datetime

def test_complex_query(agent, query_name, query, model="openai:gpt-4o"):
    """Test a complex travel query and return results"""
    print(f"\n{'='*80}")
    print(f"🧪 TESTING: {query_name}")
    print(f"📝 Query: {query}")
    print(f"{'='*80}")
    
    start_time = datetime.now()
    
    try:
        result = agent.run_complex_planning(query, model)
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        # Print summary
        print(f"\n✅ TEST COMPLETED SUCCESSFULLY")
        print(f"⏱️  Total Time: {execution_time:.2f} seconds")
        print(f"🔢 Total Steps: {result['total_steps']}")
        
        completed = len([s for s in result["planning_steps"] if s["status"] == "completed"])
        failed = len([s for s in result["planning_steps"] if s["status"] == "failed"])
        
        print(f"✅ Completed: {completed}")
        print(f"❌ Failed: {failed}")
        
        if failed > 0:
            print("Failed steps:")
            for step in result["planning_steps"]:
                if step["status"] == "failed":
                    print(f"  - {step['description']}")
        
        # Save detailed results
        result_file = f"test_results_{query_name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        print(f"💾 Detailed results saved to: {result_file}")
        
        # Print first 500 characters of the itinerary
        print(f"\n📋 SAMPLE ITINERARY (first 500 chars):")
        print("-" * 60)
        print(result["final_itinerary"][:500] + "..." if len(result["final_itinerary"]) > 500 else result["final_itinerary"])
        
        return True
        
    except Exception as e:
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        print(f"\n❌ TEST FAILED")
        print(f"⏱️  Time before failure: {execution_time:.2f} seconds")
        print(f"🚫 Error: {str(e)}")
        return False

def main():
    """Run comprehensive tests of complex travel queries"""
    
    # Load environment variables
    if not load_dotenv(".env"):
        print("❌ Failed to load .env file. Make sure API keys are configured.")
        return
    
    # Initialize agent
    agent = TravelAgent()
    
    print("🌍 COMPLEX TRAVEL AGENT TESTING SUITE")
    print("🚀 Testing 20-30 step planning with deep research features")
    print("=" * 80)
    
    # Define test scenarios
    test_scenarios = [
        {
            "name": "Multi-City European Adventure",
            "query": "Plan a 2-week adventure trip through Europe visiting Paris, Amsterdam, Berlin, and Prague in March 2024 for 2 people with a mid-range budget, interested in art, history, and local food experiences."
        },
        {
            "name": "Family Vacation Japan",
            "query": "Plan a 10-day family vacation to Japan with 2 adults and 2 children (ages 8 and 12) in October 2024, focusing on culture, technology, and kid-friendly activities with a budget of $8000 total."
        },
        {
            "name": "Solo Backpacking Southeast Asia",
            "query": "Plan a 3-week solo backpacking trip through Thailand, Vietnam, and Cambodia for a budget-conscious traveler in December 2024, focusing on local experiences, street food, and adventure activities."
        },
        {
            "name": "Luxury African Safari",
            "query": "Plan a 12-day luxury safari experience in Kenya and Tanzania for a couple's anniversary in August 2024, including wildlife viewing, cultural experiences, and romantic dining with no budget constraints."
        },
        {
            "name": "Business Trip with Leisure Extension",
            "query": "Plan a business trip to Singapore with a 4-day leisure extension exploring Malaysia and Indonesia in November 2024, focusing on efficient travel, good hotels, and weekend cultural activities."
        }
    ]
    
    # Run tests
    results = []
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n🔄 Running Test {i}/{len(test_scenarios)}")
        
        success = test_complex_query(
            agent, 
            scenario["name"], 
            scenario["query"]
        )
        
        results.append({
            "scenario": scenario["name"],
            "success": success,
            "query": scenario["query"]
        })
        
        # Brief pause between tests
        import time
        time.sleep(2)
    
    # Final summary
    print(f"\n{'='*80}")
    print("🏁 TESTING COMPLETE - FINAL SUMMARY")
    print(f"{'='*80}")
    
    successful = sum(1 for r in results if r["success"])
    total = len(results)
    
    print(f"✅ Successful Tests: {successful}/{total}")
    print(f"📊 Success Rate: {(successful/total)*100:.1f}%")
    
    if successful < total:
        print(f"\n❌ Failed Tests:")
        for result in results:
            if not result["success"]:
                print(f"  - {result['scenario']}")
    
    # Save overall summary
    summary_file = f"test_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_tests": total,
            "successful_tests": successful,
            "success_rate": (successful/total)*100,
            "results": results
        }, f, indent=2)
    
    print(f"💾 Summary saved to: {summary_file}")
    
    if successful == total:
        print(f"\n🎉 ALL TESTS PASSED! The complex travel agent is working correctly.")
    else:
        print(f"\n⚠️  Some tests failed. Check individual result files for details.")

if __name__ == "__main__":
    main()