import time
from typing import List


def mock_enterprise_search(query: str) -> List[dict]:
    """Simulate external enterprise search API with Pune weather information."""
    time.sleep(0.5)  # Simulate network delay
    
    # Pune weather information (mock data)
    pune_weather_info = [
        {
            "source": "external_weather_api",
            "content": "Pune, Maharashtra: Current weather conditions show a pleasant climate with temperatures ranging from 18°C to 30°C. The city experiences moderate humidity levels and occasional rainfall during monsoon season (June to September).",
            "score": 0.7
        },
        {
            "source": "external_weather_api",
            "content": "Pune weather forecast: The city enjoys a semi-arid climate with three distinct seasons - summer (March to May), monsoon (June to September), and winter (October to February). Average annual rainfall is approximately 722mm.",
            "score": 0.65
        },
        {
            "source": "external_weather_api",
            "content": "Pune climate data: Located at an elevation of 560 meters above sea level, Pune benefits from its proximity to the Western Ghats, resulting in cooler temperatures compared to other cities in Maharashtra. The city is known for its pleasant weather throughout most of the year.",
            "score": 0.6
        }
    ]
    
    return pune_weather_info


def mock_expert_escalation(query: str, answer: str, feedback: str) -> None:
    """Simulate expert escalation webhook."""
    print(f"\n{'='*60}")
    print(f"[ESCALATION TRIGGERED]")
    print(f"{'='*60}")
    print(f"Query: {query}")
    print(f"Answer: {answer}")
    print(f"Feedback: {feedback}")
    print(f"{'='*60}\n")
    # In a real system, would POST to external API

