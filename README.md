# PlatePal: AI-Powered Recipe & Plating Visual Generator

> **SE-B | i22582, i221513**
> Neural Recipe Generation + Conditional Image Synthesis

---

## 🧠 Architecture Overview

PlatePal combines two generative pipelines:

| Component | Model | Purpose |
|-----------|-------|---------|
| **Text Generation** | Fine-tuned DistilGPT-2 (124M params) | Structured recipe generation from ingredients |
| **Image Synthesis** | Conditional DCGAN (64→256px) | Food plating visualization from recipe text |
| **Semantic Bridge** | OpenAI CLIP (ViT-B/32) | Text→embedding conditioning for the GAN |

---

## 📂 Project Structure

```
PlatePal/
├── app/                          # Core application
│   ├── main.py                   # FastAPI backend (endpoints)
│   ├── models/
│   │   ├── text_gen.py           # RecipeGenerator (GPT-2)
│   │   └── image_gen.py          # Generator + Discriminator (DCGAN)
│   └── utils/
│       ├── data_preprocessing.py # Dataset loading & formatting
│       ├── clip_embedder.py      # CLIP embedding extraction
│       └── evaluation.py         # Perplexity, BLEU, ROUGE, FID
├── frontend/
│   └── app.py                    # Streamlit UI
├── train_text_model.py           # GPT-2 fine-tuning script
├── train_image_model.py          # DCGAN training script
├── run_evaluation.py             # Evaluation metrics runner
├── download_datasets.py          # Dataset acquisition helper
├── requirements.txt              # Python dependencies
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```

### 2. Download Datasets

```bash
python download_datasets.py --all
```

| Dataset | Size | Source |
|---------|------|--------|
| RecipeNLG | ~2.2M recipes | [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/recipenlg) |
| Food-101 | ~11K images (64x64) | [ETH Zurich](https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz) |
| CAFD | ~16.5K images (42 classes) | Kaggle |

### 3. Preprocess Text Data

```bash
python -m app.utils.data_preprocessing \
    --recipe-csv data/raw/RecipeNLG_dataset.csv \
    --output-dir data/processed
```

### 4. Train Models

**Text Model (GPT-2):**
```bash
python train_text_model.py \
    --corpus data/processed/recipes_train.csv \
    --output checkpoints/text_model \
    --epochs 5 --batch-size 32 --lr 2e-4 --fp16
```

**Image Model (DCGAN):**
```bash
python train_image_model.py \
    --food101-h5 data/raw/food_c101_n10099_r64x64x3.h5 \
    --cafd-dir data/raw/CAFD \
    --output checkpoints/dcgan \
    --epochs 200 --batch-size 128 --fp16
```

### 5. Run the Application

```bash
# Terminal 1: Backend
uvicorn app.main:app --reload --port 8000

# Terminal 2: Frontend
streamlit run frontend/app.py
```

### 6. Evaluate

```bash
python run_evaluation.py \
    --text-model checkpoints/text_model/best \
    --image-model checkpoints/dcgan/generator_final.pth
```

---

## 📊 Evaluation Metrics

| Metric | Type | Target |
|--------|------|--------|
| **Perplexity** | Text (fluency) | Lower is better |
| **BLEU-4** | Text (n-gram overlap) | Higher is better |
| **ROUGE-L** | Text (sequential logic) | Higher is better |
| **Ingredient Coverage** | Text (completeness) | ≥ 80% |
| **FID** | Image (distribution quality) | Lower is better |

---

## ⚙️ Hardware Recommendations

Optimized for **NVIDIA RTX 5060 Ti (16GB)**:

- DCGAN batch size: `128`
- Transformer batch size: `32`
- Mixed precision (FP16): enabled via `--fp16`
- Dataset format: HDF5 for zero-overhead loading

---

## 🔧 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/generate` | Full pipeline (recipe + image) |
| `POST` | `/generate/text` | Recipe generation only |
| `POST` | `/generate/image` | Image generation only |

**Example request:**
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"ingredients": "chicken, rice, garlic, cumin"}'
```
