import sys
from pathlib import Path
import os

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.utils.recipe_db import get_semantic_context

def test_rag():
    print("--- PlatePal: RAG Retrieval Test ---")
    
    test_queries = [
        "spicy chicken wings",
        "creamy pasta with mushrooms",
        "healthy salmon salad",
        "traditional beef stew"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        context = get_semantic_context(query, top_k=2)
        if context:
            print(context)
        else:
            print("No context retrieved. Is the index built?")

if __name__ == "__main__":
    test_rag()
