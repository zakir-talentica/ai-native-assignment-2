"""
RAGAS Evaluation Script for RAG PoC

This script evaluates the RAG system using RAGAS metrics.
Note: Install ragas and datasets packages:
    pip install ragas datasets python-dotenv langchain-openai
"""

import json
import os
import sys
from typing import List, Dict

# Add parent directory to path to import backend modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Try loading from root directory first, then evaluation directory
    load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
    load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
except ImportError:
    print("Warning: python-dotenv not installed. Install with: pip install python-dotenv")
    print("Make sure OPENAI_API_KEY is set as an environment variable.")

# Verify API key is set
if not os.getenv('OPENAI_API_KEY'):
    print("="*60)
    print("ERROR: OPENAI_API_KEY not found!")
    print("="*60)
    print("Please either:")
    print("1. Set OPENAI_API_KEY as an environment variable, or")
    print("2. Create a .env file in the project root with:")
    print("   OPENAI_API_KEY=your_api_key_here")
    print("\nYou can copy env.example to .env and update it.")
    sys.exit(1)

try:
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
    from datasets import Dataset
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
except ImportError as e:
    print("Error: Please install required packages:")
    print("  pip install ragas datasets python-dotenv langchain-openai")
    print(" error: ", e)
    sys.exit(1)


def load_test_queries(file_path: str = "test_queries.json") -> List[Dict]:
    """Load test queries from JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {file_path} not found. Using default queries.")
        return [
            {
                "query": "What is retrieval-augmented generation?",
                "expected_keywords": ["retrieval", "generation", "LLM"]
            },
            {
                "query": "Explain vector databases",
                "expected_keywords": ["embeddings", "similarity", "search"]
            }
        ]


def prepare_evaluation_dataset(
    questions: List[str],
    answers: List[str],
    contexts: List[List[str]],
    ground_truths: List[str] = None
) -> Dataset:
    """Prepare dataset in RAGAS format."""
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts
    }
    
    if ground_truths:
        data["ground_truth"] = ground_truths
    
    return Dataset.from_dict(data)


def run_evaluation(dataset: Dataset, metrics: List = None) -> dict:
    """Run RAGAS evaluation on the dataset."""
    # Configure embeddings and LLM for RAGAS
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=os.getenv('OPENAI_API_KEY'))
    llm = ChatOpenAI(model="gpt-4.1", openai_api_key=os.getenv('OPENAI_API_KEY'))
    
    if metrics is None:
        # Use default metrics
        metrics = [
            faithfulness,          # How factual is the answer based on context
            answer_relevancy,      # How relevant is the answer to the question
            context_recall,        # How much of ground truth is in retrieved context
            context_precision      # How relevant are retrieved contexts
        ]
    
    print("Running RAGAS evaluation...")
    print(f"Metrics: {[m.name for m in metrics]}")
    print(f"Dataset size: {len(dataset)}")
    
    results = evaluate(
        dataset,
        metrics=metrics,
        embeddings=embeddings,
        llm=llm
    )
    
    return results


def main():
    """Main evaluation function."""
    print("="*60)
    print("RAG PoC - RAGAS Evaluation")
    print("="*60)
    
    # Example: Prepare a sample evaluation dataset
    # In a real scenario, you would:
    # 1. Query your RAG system for each test question
    # 2. Collect the answers and retrieved contexts
    # 3. Compare against ground truth if available
    
    sample_questions = [
        "What is retrieval-augmented generation?",
        "How does FAISS work?",
        "What are embeddings?"
    ]
    
    # These would come from your RAG system
    sample_answers = [
        "Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval with text generation. It retrieves relevant documents and uses them as context for generating accurate responses.",
        "FAISS (Facebook AI Similarity Search) is a library for efficient similarity search and clustering of dense vectors. It enables fast nearest neighbor search in high-dimensional spaces.",
        "Embeddings are numerical vector representations of text that capture semantic meaning. Similar texts have similar embeddings, enabling semantic search."
    ]
    
    # Retrieved contexts for each question
    sample_contexts = [
        [
            "RAG combines retrieval and generation to provide accurate, context-aware responses.",
            "The retrieval component fetches relevant documents, which the generation model uses as context."
        ],
        [
            "FAISS is designed for efficient similarity search in large collections of vectors.",
            "It provides various indexing methods optimized for speed and memory usage."
        ],
        [
            "Text embeddings are created by encoding text into fixed-size vectors.",
            "These vectors capture semantic relationships between words and phrases."
        ]
    ]
    
    # Optional: Ground truth answers for comparison
    sample_ground_truths = [
        "RAG is a technique that retrieves relevant information and uses it to generate informed responses.",
        "FAISS is a library for fast similarity search in vector databases.",
        "Embeddings are vector representations that encode semantic meaning of text."
    ]
    
    # Prepare dataset
    dataset = prepare_evaluation_dataset(
        questions=sample_questions,
        answers=sample_answers,
        contexts=sample_contexts,
        ground_truths=sample_ground_truths
    )
    
    # Run evaluation
    try:
        results = run_evaluation(dataset)
        
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(results)
        print("="*60)
        print("Evaluation complete!")
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nNote: Make sure you have set OPENAI_API_KEY environment variable")
        print("RAGAS uses OpenAI models for evaluation metrics.")


if __name__ == "__main__":
    main()

