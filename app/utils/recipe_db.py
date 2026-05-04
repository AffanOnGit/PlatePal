import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class SemanticRecipeRetriever:
    """
    Upgraded RAG System: Performs Semantic Search over 20,000+ recipes.
    Uses FAISS for vector retrieval and Sentence-Transformers for semantic understanding.
    """
    def __init__(self, index_dir="data/rag_index"):
        self.index_path = os.path.join(index_dir, "recipe_index.faiss")
        self.data_path = os.path.join(index_dir, "recipe_data.pkl")
        self.model_name = 'all-MiniLM-L6-v2'
        
        self.index = None
        self.recipes = None
        self.model = None

    def _load(self):
        """Lazy load the index and model to save resources if not used."""
        if self.index is None:
            if not os.path.exists(self.index_path) or not os.path.exists(self.data_path):
                print(f"[RAG] Warning: Index files not found at {self.index_path}. Run build_rag_index.py first.")
                return False
            
            print(f"[RAG] Loading Semantic Index...")
            self.index = faiss.read_index(self.index_path)
            with open(self.data_path, 'rb') as f:
                self.recipes = pickle.load(f)
            self.model = SentenceTransformer(self.model_name)
            print(f"[RAG] Index Loaded. {len(self.recipes)} recipes active.")
        return True

    def get_pro_recipe(self, ingredients_str, top_k=3):
        """
        Retrieves the top K most similar recipes from the database.
        Returns a structured context string for the LLM.
        """
        if not self._load():
            return None

        # 1. Embed query
        query_vec = self.model.encode([ingredients_str]).astype('float32')
        
        # 2. Search FAISS
        distances, indices = self.index.search(query_vec, top_k)
        
        # 3. Format Results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.recipes):
                res = self.recipes[idx]
                results.append({
                    "title": res['title'],
                    "ingredients": res['ingredients'],
                    "instructions": res['directions']
                })
        
        # For backward compatibility with the existing main.py logic, 
        # we return the first result as the 'primary' if asked, 
        # but the LLM will actually use the whole list via a different method.
        return results[0] if results else None

    def get_rag_context(self, ingredients_str, top_k=3):
        """Returns a formatted string of real recipes to be used as LLM context."""
        if not self._load():
            return ""

        query_vec = self.model.encode([ingredients_str]).astype('float32')
        distances, indices = self.index.search(query_vec, top_k)
        
        context = "--- REAL RECIPE CONTEXT FROM DATABASE ---\n"
        for i, idx in enumerate(indices[0]):
            if idx < len(self.recipes):
                r = self.recipes[idx]
                context += f"\nRECIPE #{i+1}: {r['title']}\n"
                context += f"Ingredients Found: {r['ingredients']}\n"
                context += f"Techniques: {r['directions'][:300]}...\n"
        
        return context

# Singleton instance for the app
retriever = SemanticRecipeRetriever()

def get_pro_recipe(ingredients_str):
    """Bridge function for existing main.py logic."""
    return retriever.get_pro_recipe(ingredients_str)

def get_semantic_context(ingredients_str, top_k=3):
    """New function for LLM context injection."""
    return retriever.get_rag_context(ingredients_str, top_k=top_k)
