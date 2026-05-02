"""
PlatePal — FastAPI Backend
===========================

Serves the complete PlatePal pipeline:
  1. Receive ingredients from the frontend.
  2. Generate a structured recipe via GPT-2.
  3. Extract a CLIP embedding from the recipe text.
  4. Feed the embedding + noise into the DCGAN Generator.
  5. Return the recipe card and plating image.
"""

import os
import base64
import logging
from io import BytesIO

import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.models.text_gen import RecipeGenerator
from app.models.image_gen import Generator as ImageGenerator
from app.utils.clip_embedder import get_embedder

# ─────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────

TEXT_CHECKPOINT  = os.getenv("PLATEPAL_TEXT_CKPT",  "checkpoints/text_model/best")
IMAGE_CHECKPOINT = os.getenv("PLATEPAL_IMG_CKPT",   "checkpoints/dcgan/generator_final.pth")
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
Z_DIM            = 100
EMBED_DIM        = 512

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("platepal")

# ─────────────────────────────────────────────────────────────────────
# App Init
# ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="PlatePal API",
    description="AI-powered recipe generation and food plating visualization",
    version="1.0.0",
)

# CORS — allow Streamlit frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────
# Model Loading (lazy singletons)
# ─────────────────────────────────────────────────────────────────────

_recipe_gen: RecipeGenerator = None
_img_gen: ImageGenerator = None
_embedder = None


def get_recipe_gen() -> RecipeGenerator:
    global _recipe_gen
    if _recipe_gen is None:
        ckpt = TEXT_CHECKPOINT if os.path.isdir(TEXT_CHECKPOINT) else None
        _recipe_gen = RecipeGenerator(checkpoint_dir=ckpt, device=DEVICE)
    return _recipe_gen


def get_img_gen() -> ImageGenerator:
    global _img_gen
    if _img_gen is None:
        _img_gen = ImageGenerator(z_dim=Z_DIM, embed_dim=EMBED_DIM).to(DEVICE)
        if os.path.isfile(IMAGE_CHECKPOINT):
            _img_gen.load_state_dict(torch.load(IMAGE_CHECKPOINT, map_location=DEVICE))
            logger.info(f"Loaded DCGAN weights from {IMAGE_CHECKPOINT}")
        else:
            logger.warning(f"No DCGAN weights at {IMAGE_CHECKPOINT} — using random init")
        _img_gen.eval()
    return _img_gen


def get_clip_embedder():
    global _embedder
    if _embedder is None:
        _embedder = get_embedder(use_clip=True, device=DEVICE)
    return _embedder


# ─────────────────────────────────────────────────────────────────────
# Request / Response Schemas
# ─────────────────────────────────────────────────────────────────────

class IngredientRequest(BaseModel):
    ingredients: str
    temperature: float = 0.7
    max_length: int = 512


class PlatePalResponse(BaseModel):
    recipe: str
    image_base64: str
    ingredients_used: str


class HealthResponse(BaseModel):
    status: str
    device: str
    text_model_loaded: bool
    image_model_loaded: bool


# ─────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────

@app.get("/", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="ok",
        device=DEVICE,
        text_model_loaded=_recipe_gen is not None,
        image_model_loaded=_img_gen is not None,
    )


@app.post("/generate", response_model=PlatePalResponse)
async def generate(request: IngredientRequest):
    """
    Full PlatePal pipeline:
    ingredients → recipe text → CLIP embedding → DCGAN image
    """
    try:
        ingredients = request.ingredients.strip()
        if not ingredients:
            raise HTTPException(status_code=400, detail="Ingredients cannot be empty")

        logger.info(f"Generating for: {ingredients}")

        # ── Step 1: Recipe Generation ──────────────────────────────
        recipe_gen = get_recipe_gen()
        recipe_text = recipe_gen.generate_recipe(
            ingredients,
            max_length=request.max_length,
            temperature=request.temperature,
        )

        # ── Step 2: CLIP Embedding ─────────────────────────────────
        embedder = get_clip_embedder()
        embedding = embedder.embed_recipe(
            title=ingredients,  # use ingredients as a proxy for title
            ingredients=ingredients,
        )  # shape (1, 512)

        # ── Step 3: Image Synthesis ────────────────────────────────
        img_gen = get_img_gen()
        z = torch.randn(1, Z_DIM, device=DEVICE)

        with torch.no_grad():
            img_tensor = img_gen(z, embedding)  # (1, 3, 64, 64)

        # Rescale [-1, 1] → [0, 255]
        img_tensor = ((img_tensor + 1) / 2).clamp(0, 1)
        img_np = (img_tensor.squeeze().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        # Upscale to 256x256 for nicer display
        img_pil = Image.fromarray(img_np).resize((256, 256), Image.LANCZOS)

        buffered = BytesIO()
        img_pil.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode()

        return PlatePalResponse(
            recipe=recipe_text,
            image_base64=img_b64,
            ingredients_used=ingredients,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Generation failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/text")
async def generate_text_only(request: IngredientRequest):
    """Generate only the recipe text (no image)."""
    recipe_gen = get_recipe_gen()
    recipe = recipe_gen.generate_recipe(
        request.ingredients,
        max_length=request.max_length,
        temperature=request.temperature,
    )
    return {"recipe": recipe}


@app.post("/generate/image")
async def generate_image_only(request: IngredientRequest):
    """Generate only the plating image."""
    embedder = get_clip_embedder()
    embedding = embedder.embed_recipe(title=request.ingredients, ingredients=request.ingredients)

    img_gen = get_img_gen()
    z = torch.randn(1, Z_DIM, device=DEVICE)

    with torch.no_grad():
        img_tensor = img_gen(z, embedding)

    img_tensor = ((img_tensor + 1) / 2).clamp(0, 1)
    img_np = (img_tensor.squeeze().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_np).resize((256, 256), Image.LANCZOS)

    buffered = BytesIO()
    img_pil.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()

    return {"image_base64": img_b64}
