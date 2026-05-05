import os
from groq import Groq

class GroqRecipeGenerator:
    """
    High-fidelity recipe generation using the Groq API.
    Replaces hallucinating local models with industry-standard LLMs.
    """
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            print("[Warning] GROQ_API_KEY not found. LLM generation will fail.")
        
        self.client = Groq(api_key=self.api_key) if self.api_key else None
        self.model = "llama-3.3-70b-versatile"

    def generate_recipe(self, ingredients: str, context: str = "", max_length: int = 512, temperature: float = 0.5) -> str:
        """
        Generates a professional recipe using Llama 3 via Groq.
        Augments the prompt with Semantic RAG context if provided.
        """
        system_prompt = (
            "You are a Michelin-star Executive Chef. "
            "Generate a structured recipe based on the provided ingredients. "
        )
        
        if context:
            system_prompt += (
                "\nUSE THE FOLLOWING REAL RECIPES AS TECHNICAL CONTEXT:\n"
                f"{context}\n"
                "\nINSTRUCTIONS: If the context recipes are relevant, use their flavor profiles "
                "and culinary techniques. If they are not relevant, use your own expertise. "
            )

        system_prompt += (
            "\nYou MUST use this exact format:\n"
            "<TITLE_START> [Dish Name] <INPUT_START> [Ingredients List] <INSTR_START> [Step-by-step instructions]\n"
            "Keep the tone professional, concise, and focused on high-end plating."
        )

        if not self.client:
            return "<TITLE_START> API Key Error <INPUT_START> N/A <INSTR_START> Please set your GROQ_API_KEY in the .env file to enable recipe generation."

        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Ingredients: {ingredients}"}
                ],
                model=self.model,
                temperature=0.3, # Low temperature for accurate cooking
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            print(f"[Groq-Error] Falling back to local model: {e}")
            return f"<TITLE_START> Chef's Special <INPUT_START> {ingredients} <INSTR_START> Error connecting to Groq. Please check your API key."
