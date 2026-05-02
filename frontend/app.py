"""
PlatePal — Streamlit Frontend
===============================

A single-page application that displays:
  - An ingredient input area
  - A generated recipe card (left column)
  - A generated plating visual (right column)
"""

import streamlit as st
import requests
import base64
from PIL import Image
from io import BytesIO

# ─────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="PlatePal — AI Recipe & Plating Generator",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────
# Custom CSS for premium look
# ─────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
    }

    .sub-header {
        text-align: center;
        color: #888;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    .recipe-card {
        background: #1a1a2e;
        border-radius: 16px;
        padding: 2rem;
        border: 1px solid #333;
        color: #e0e0e0;
        line-height: 1.7;
    }

    .recipe-card h1, .recipe-card h2 {
        color: #667eea;
    }

    .image-card {
        background: #1a1a2e;
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid #333;
        text-align: center;
    }

    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        width: 100%;
        transition: transform 0.2s, box-shadow 0.2s;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }

    .metric-box {
        background: #16213e;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #333;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────

st.markdown('<p class="main-header">🍽️ PlatePal</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Transform your random ingredients into a gourmet experience — powered by AI</p>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/chef-hat.png", width=80)
    st.title("⚙️ Settings")

    backend_url = st.text_input(
        "Backend API URL",
        value="http://localhost:8000",
        help="The FastAPI server address"
    )

    st.divider()

    st.subheader("Generation Parameters")
    temperature = st.slider("Creativity (Temperature)", 0.1, 1.5, 0.7, 0.1,
                            help="Higher = more creative, lower = more predictable")
    max_length = st.slider("Max Recipe Length (tokens)", 128, 1024, 512, 64)

    st.divider()

    st.subheader("Quick Presets")
    preset = st.selectbox("Try a preset:", [
        "Custom",
        "🇵🇰 Pakistani Biryani",
        "🇮🇹 Italian Pasta",
        "🇯🇵 Japanese Bowl",
        "🇲🇽 Mexican Tacos",
        "🥗 Healthy Salad",
    ])

    preset_map = {
        "🇵🇰 Pakistani Biryani": "basmati rice, chicken, onion, yogurt, ginger, garlic, biryani masala, saffron, ghee",
        "🇮🇹 Italian Pasta": "spaghetti, tomato, garlic, basil, parmesan, olive oil, chili flakes",
        "🇯🇵 Japanese Bowl": "short-grain rice, salmon, soy sauce, sesame seeds, avocado, nori, wasabi",
        "🇲🇽 Mexican Tacos": "corn tortillas, ground beef, cheddar, lettuce, tomato, sour cream, cumin",
        "🥗 Healthy Salad": "quinoa, spinach, avocado, cherry tomatoes, feta, lemon, olive oil",
    }

    st.divider()
    st.caption("Built by SE-B i22582, i221513")
    st.caption("DistilGPT-2 + DCGAN + CLIP")

# ─────────────────────────────────────────────────────────────────────
# Main Input Area
# ─────────────────────────────────────────────────────────────────────

default_ingredients = preset_map.get(preset, "")

col_input, col_btn = st.columns([4, 1])

with col_input:
    ingredients = st.text_area(
        "🥘 Enter your ingredients (comma-separated):",
        value=default_ingredients,
        height=100,
        placeholder="e.g., chicken, rice, garlic, soy sauce, ginger",
    )

with col_btn:
    st.write("")  # spacing
    st.write("")
    generate_clicked = st.button("🚀 Generate", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────
# Generation & Display
# ─────────────────────────────────────────────────────────────────────

if generate_clicked:
    if not ingredients.strip():
        st.warning("⚠️ Please enter at least a few ingredients first!")
    else:
        with st.spinner("🍳 Generating your recipe and plating visual... this may take a moment."):
            try:
                response = requests.post(
                    f"{backend_url}/generate",
                    json={
                        "ingredients": ingredients.strip(),
                        "temperature": temperature,
                        "max_length": max_length,
                    },
                    timeout=120,
                )

                if response.status_code == 200:
                    data = response.json()
                    recipe_text = data["recipe"]
                    img_data = data["image_base64"]

                    st.success("✅ Recipe generated successfully!")
                    st.divider()

                    col1, col2 = st.columns([3, 2])

                    with col1:
                        st.markdown("### 📜 Your Custom Recipe")
                        st.markdown(f'<div class="recipe-card">{recipe_text}</div>',
                                    unsafe_allow_html=True)

                    with col2:
                        st.markdown("### 🖼️ Suggested Plating")
                        img_bytes = base64.b64decode(img_data)
                        img = Image.open(BytesIO(img_bytes))
                        st.image(img, use_container_width=True, caption="AI-generated plating suggestion")

                    # Ingredient coverage indicator
                    st.divider()
                    ing_list = [i.strip().lower() for i in ingredients.split(",")]
                    recipe_lower = recipe_text.lower()
                    found = [ing for ing in ing_list if ing in recipe_lower]
                    coverage = len(found) / max(len(ing_list), 1)

                    mcol1, mcol2, mcol3 = st.columns(3)
                    with mcol1:
                        st.metric("🧪 Ingredient Coverage", f"{coverage:.0%}")
                    with mcol2:
                        st.metric("📝 Recipe Length", f"{len(recipe_text.split())} words")
                    with mcol3:
                        st.metric("🎯 Ingredients Used", f"{len(found)}/{len(ing_list)}")

                else:
                    st.error(f"❌ Server returned {response.status_code}: {response.text}")

            except requests.exceptions.ConnectionError:
                st.error(
                    "🔌 Could not connect to the backend. "
                    "Make sure the FastAPI server is running:\n\n"
                    "```bash\nuvicorn app.main:app --reload --port 8000\n```"
                )
            except requests.exceptions.Timeout:
                st.error("⏱️ Request timed out. The model may still be loading.")
            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")

# ─────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────

st.divider()
st.markdown(
    """
    <div style="text-align: center; color: #666; font-size: 0.85rem;">
        <strong>PlatePal v1.0</strong> — Neural Recipe & Plating Generator<br>
        Powered by DistilGPT-2 · DCGAN · CLIP · FastAPI · Streamlit<br>
        SE-B | i22582, i221513
    </div>
    """,
    unsafe_allow_html=True,
)
