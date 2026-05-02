import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os

class StableDiffusionGenerator:
    """
    PlatePal Image Generator — Powered by Stable Diffusion
    =====================================================
    Replaces the legacy CVAE for photorealistic, high-end food plating.
    Uses 'stable-diffusion-v1-5' with professional food-photography prompts.
    """

    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", device="cuda"):
        self.device = device
        print(f"[SD-Generator] Loading {model_id} on {device}...")
        
        # Load pipeline with FP16 for speed and lower VRAM usage
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.float16,
            safety_checker=None # Disabled for food-only speed, can be enabled
        )
        self.pipe = self.pipe.to(device)
        
        # Enable memory optimizations for 8GB VRAM cards (RTX 5060 Ti)
        self.pipe.enable_attention_slicing()
        
        print("[SD-Generator] Model ready. Loading Food LoRA...")
        
        # Load YOUR Custom-Trained Food LoRA
        local_lora = "checkpoints/food_lora"
        try:
            if os.path.exists(local_lora):
                self.pipe.load_lora_weights(local_lora)
                print(f"[SD-Generator] SUCCESS: Custom PlatePal LoRA Active.")
            else:
                print(f"[SD-Generator] Warning: Local LoRA not found at {local_lora}. Using base model.")
        except Exception as e:
            print(f"[SD-Generator] Error loading custom LoRA: {e}")

    def generate(self, dish_name: str, ingredients: str) -> Image:
        """
        Generates a high-end food plating image.
        Wraps inputs in a 'Michelin-Star' prompt.
        """
        # Professional Prompt Engineering
        prompt = (
            f"Professional food photography of {dish_name}, "
            f"featuring {ingredients}, michelin star plating, "
            f"luxury ceramic plate, high detail, 8k, macro lens, "
            f"soft studio lighting, bokeh background, appetizer style, "
            f"highly appetizing, vivid colors"
        )
        
        negative_prompt = (
            "blurry, low quality, distorted, messy, plastic, "
            "unappetizing, dark, noisy, text, watermark, logo, "
            "cluttered background, simple home cooking"
        )

        print(f"[SD-Generator] Generating: {dish_name}")
        
        # Generate with 25 steps (fast but high quality for SD v1.5)
        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=30,
            guidance_scale=8.0
        ).images[0]
        
        return image

    def load_lora(self, lora_path: str):
        """Optionally load a food-specific LoRA for even better textures."""
        if os.path.exists(lora_path):
            print(f"[SD-Generator] Loading LoRA from {lora_path}")
            self.pipe.load_lora_weights(lora_path)
