"""
PlatePal — Recipe Text Generator
==================================

Wraps a fine-tuned GPT-2 / DistilGPT-2 for structured recipe generation.
Supports loading from a local checkpoint or falling back to the base model.
"""

import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Default special tokens (must match the training script)
SPECIAL_TOKENS = {
    "bos_token": "<RECIPE_START>",
    "eos_token": "<RECIPE_END>",
    "pad_token": "<PAD>",
    "additional_special_tokens": [
        "<INPUT_START>", "<INSTR_START>", "<TITLE>",
    ],
}


class RecipeGenerator:
    """
    Generates structured recipes from ingredient lists using
    an autoregressive language model.
    """

    def __init__(
        self,
        model_name: str = "distilgpt2",
        checkpoint_dir: str = None,
        device: str = None,
    ):
        """
        Args:
            model_name: HuggingFace model ID (used if no checkpoint is given).
            checkpoint_dir: Path to a fine-tuned checkpoint directory
                            (contains config.json, pytorch_model.bin, etc.).
            device: 'cuda' or 'cpu'. Auto-detected if None.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        load_from = checkpoint_dir if (checkpoint_dir and os.path.isdir(checkpoint_dir)) else model_name

        print(f"[RecipeGenerator] Loading tokenizer from {load_from}")
        self.tokenizer = GPT2Tokenizer.from_pretrained(load_from)

        # Ensure special tokens are present (safe to call even if already added)
        num_added = self.tokenizer.add_special_tokens(SPECIAL_TOKENS)

        print(f"[RecipeGenerator] Loading model from {load_from}")
        self.model = GPT2LMHeadModel.from_pretrained(load_from).to(self.device)

        if num_added > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.model.eval()
        print(f"[RecipeGenerator] Ready on {self.device}  "
              f"(vocab: {len(self.tokenizer):,}, params: {sum(p.numel() for p in self.model.parameters()):,})")

    # ─────────────────────────────────────────────────────────────
    # Generation
    # ─────────────────────────────────────────────────────────────

    def generate_recipe(
        self,
        ingredients: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
        num_return: int = 1,
    ) -> str:
        """
        Generate a recipe from a comma-separated ingredient list.

        Returns:
            Decoded recipe string (with special tokens stripped for display).
        """
        prompt = f"<RECIPE_START><INPUT_START>{ingredients}<INSTR_START>"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                num_return_sequences=num_return,
                no_repeat_ngram_size=3,
                do_sample=True,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        raw = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        return self._format_output(raw)

    # ─────────────────────────────────────────────────────────────
    # Output Formatting
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _format_output(raw: str) -> str:
        """
        Clean the raw model output into a human-readable recipe card.
        """
        # Remove leading/trailing control tokens
        text = raw
        for token in ["<RECIPE_START>", "<RECIPE_END>", "<PAD>"]:
            text = text.replace(token, "")

        # Split into sections
        parts = {}
        if "<TITLE>" in text:
            title_rest = text.split("<TITLE>", 1)[1]
            if "<INPUT_START>" in title_rest:
                parts["title"] = title_rest.split("<INPUT_START>")[0].strip()
            else:
                parts["title"] = title_rest.strip()

        if "<INPUT_START>" in text:
            ing_rest = text.split("<INPUT_START>", 1)[1]
            if "<INSTR_START>" in ing_rest:
                parts["ingredients"] = ing_rest.split("<INSTR_START>")[0].strip()
            else:
                parts["ingredients"] = ing_rest.strip()

        if "<INSTR_START>" in text:
            parts["instructions"] = text.split("<INSTR_START>", 1)[1].strip()

        # Build formatted output
        lines = []
        if "title" in parts:
            lines.append(f"# {parts['title']}\n")
        if "ingredients" in parts:
            lines.append("## Ingredients")
            for ing in parts["ingredients"].split(","):
                ing = ing.strip()
                if ing:
                    lines.append(f"- {ing}")
            lines.append("")
        if "instructions" in parts:
            lines.append("## Instructions")
            lines.append(parts["instructions"])

        return "\n".join(lines) if lines else text


if __name__ == "__main__":
    gen = RecipeGenerator(device="cpu")
    result = gen.generate_recipe("chicken, rice, spinach, garlic, lemon")
    print(result)
