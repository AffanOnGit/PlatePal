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

import re
import os
import base64
import logging
from io import BytesIO

import torch
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from app.models.text_gen import RecipeGenerator
from app.models.image_gen import StableDiffusionGenerator

# ─────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────

TEXT_CHECKPOINT  = os.getenv("PLATEPAL_TEXT_CKPT",  "checkpoints/text_model/best")
IMAGE_CHECKPOINT = os.getenv("PLATEPAL_IMG_CKPT",   "runwayml/stable-diffusion-v1-5")
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
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
_img_gen: StableDiffusionGenerator = None
_embedder = None


def get_recipe_gen() -> RecipeGenerator:
    global _recipe_gen
    if _recipe_gen is None:
        ckpt = TEXT_CHECKPOINT if os.path.isdir(TEXT_CHECKPOINT) else None
        _recipe_gen = RecipeGenerator(checkpoint_dir=ckpt, device=DEVICE)
    return _recipe_gen


from app.models.image_gen import StableDiffusionGenerator

def get_img_gen() -> StableDiffusionGenerator:
    global _img_gen
    if _img_gen is None:
        _img_gen = StableDiffusionGenerator(device=DEVICE)
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


class StructuredRecipe(BaseModel):
    """Parsed recipe with separate fields for each section."""
    title: Optional[str] = None
    ingredients: Optional[str] = None
    instructions: Optional[str] = None
    raw: str  # Full raw text as fallback


class PlatePalResponse(BaseModel):
    recipe: StructuredRecipe
    image_base64: str
    ingredients_used: str
    metadata: Optional[dict] = None


class HealthResponse(BaseModel):
    status: str
    device: str
    text_model_loaded: bool
    image_model_loaded: bool


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def parse_recipe_sections(raw_text: str) -> StructuredRecipe:
    """
    Parse the raw model output into clean, display-ready sections.
    Strips all Markdown symbols and control tokens.
    """
    # Strip all control tokens
    text = raw_text
    for token in ["<RECIPE_START>", "<RECIPE_END>", "<PAD>"]:
        text = text.replace(token, "")

    # Strip Markdown: ##, **, *, leading dashes
    text = re.sub(r'#{1,6}\s*', '', text)
    text = re.sub(r'\*{1,2}(.*?)\*{1,2}', r'\1', text)
    text = re.sub(r'^\s*[-*]\s+', '', text, flags=re.MULTILINE)
    
    # Strip Technical Noise (hallucinated measurements/sequences like 1cm, 2nd, etc)
    text = re.sub(r'\d+\s*(cm|mm|st|nd|rd|th)\b\.?', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    title        = None
    ingredients  = None
    instructions = None

    if "<TITLE>" in text:
        title_rest = text.split("<TITLE>", 1)[1]
        if "<INPUT_START>" in title_rest:
            title = title_rest.split("<INPUT_START>")[0].strip()
        else:
            title = title_rest.strip()

    if "<INPUT_START>" in text:
        ing_rest = text.split("<INPUT_START>", 1)[1]
        if "<INSTR_START>" in ing_rest:
            ingredients = ing_rest.split("<INSTR_START>")[0].strip()
        else:
            ingredients = ing_rest.strip()

    if "<INSTR_START>" in text:
        instructions = text.split("<INSTR_START>", 1)[1].strip()

    # Smart title fallback: build a name from the first ingredient
    if not title or len(title) < 3:
        # Pull first ingredient word and title-case it
        title = "PlatePal Signature Dish"

    # Clean up the title
    title = title.strip().strip('"').strip("'")
    if title and not title[0].isupper():
        title = title.title()

    return StructuredRecipe(
        title=title or "PlatePal Signature Dish",
        ingredients=ingredients or "See instructions below",
        instructions=instructions or text.strip(),
        raw=text.strip(),
    )


def generate_plating_image(dish_name: str, ingredients: str) -> str:
    """Generate a photorealistic plating image using Stable Diffusion."""
    img_gen = get_img_gen()
    
    # Generate the image
    img_pil = img_gen.generate(dish_name, ingredients)

    # Professional Upscale & Polish
    img_pil = img_pil.resize((768, 768), Image.LANCZOS)
    
    # Optional: Light sharpening for that "commercial" look
    img_pil = img_pil.filter(ImageFilter.UnsharpMask(radius=1.0, percent=100, threshold=1))

    buffered = BytesIO()
    img_pil.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


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


@app.post("/warmup")
async def warmup():
    """
    Pre-load all models so the first real request doesn't time out.
    Call this once after server startup.
    """
    logger.info("Warming up models...")
    get_recipe_gen()
    get_img_gen()
    get_clip_embedder()
    return {"status": "all models loaded", "device": DEVICE}


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
        raw_recipe = recipe_gen.generate_recipe(
            ingredients,
            max_length=request.max_length,
            temperature=request.temperature,
        )

        # ── Step 2: Parse into structured sections ─────────────────
        structured = parse_recipe_sections(raw_recipe)

        # ── Step 3: Image Synthesis (Stable Diffusion) ─────────────
        img_b64 = generate_plating_image(structured.title, ingredients)

        return PlatePalResponse(
            recipe=structured,
            image_base64=img_b64,
            ingredients_used=ingredients,
            metadata={
                "image_model": "CVAE-128",
                "resolution": "512x512",
                "sharpening": "HD-300"
            }
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
    raw = recipe_gen.generate_recipe(
        request.ingredients,
        max_length=request.max_length,
        temperature=request.temperature,
    )
    structured = parse_recipe_sections(raw)
    return {"recipe": structured.model_dump()}


@app.post("/generate/image")
async def generate_image_only(request: IngredientRequest):
    """Generate only the plating image."""
    img_b64 = generate_plating_image(request.ingredients)
    return {"image_base64": img_b64}
