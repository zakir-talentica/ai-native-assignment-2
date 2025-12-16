import time
from typing import List


def mock_enterprise_search(query: str) -> List[dict]:
    """Simulate external enterprise search API."""
    time.sleep(0.5)  # Simulate network delay
    return [
        {
            "source": "external_system",
            "content": f"Mock external result 1 for query: {query}",
            "score": 0.6
        },
        {
            "source": "external_system",
            "content": f"Mock external result 2 for query: {query}",
            "score": 0.5
        }
    ]


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

