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
        # Start with TITLE_START so it generates a name first
        prompt = f"<RECIPE_START><TITLE_START>"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # We also want to provide the ingredients as context
        # But for this model's specific training, we usually do:
        # <RECIPE_START><INPUT_START>ingredients<TITLE_START>
        prompt = f"<RECIPE_START><INPUT_START>{ingredients}<TITLE_START>"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=150,           # Shorter is safer
                num_return_sequences=num_return,
                no_repeat_ngram_size=3,
                repetition_penalty=1.6,   # Stronger penalty for nonsense
                do_sample=True,
                top_k=40,
                top_p=0.85,               # More focused sampling
                temperature=0.4,          # Even colder for less "creativity"
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        raw = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Strip out any random Unicode/garbage characters like \UooB
        raw = "".join(i for i in raw if ord(i) < 128)
        
        # --- RAMBLING FILTER ---
        # If the AI starts talking about things that aren't cooking, cut it off.
        nonsense_keywords = ["Christmas", "Holiday", "Interview", "Post", "Blog", "Author", "Subscribe", "Email", "Thai", "Japan", "Visit"]
        for word in nonsense_keywords:
            if word in raw:
                raw = raw.split(word)[0]
        
        # If it generated an <|endoftext|> token, stop there
        if "<|endoftext|>" in raw:
            raw = raw.split("<|endoftext|>")[0]

        return raw
