import os
from groq import Groq

class GroqRecipeGenerator:
    """
    High-fidelity recipe generation using the Groq API.
    Replaces hallucinating local models with industry-standard LLMs.
    """
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.3-70b-versatile" # Updated to latest supported model

    def generate_recipe(self, ingredients: str, max_length: int = 512, temperature: float = 0.5) -> str:
        """
        Generates a professional recipe using Llama 3 via Groq.
        Formats the output specifically for the PlatePal parser.
        """
        system_prompt = (
            "You are a Michelin-star Executive Chef. "
            "Generate a structured recipe based on the provided ingredients. "
            "You MUST use this exact format:\n"
            "<TITLE_START> [Dish Name] <INPUT_START> [Ingredients List] <INSTR_START> [Step-by-step instructions]\n"
            "Keep the tone professional, concise, and focused on high-end plating."
        )

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
