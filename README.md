# PlatePal: AI-Powered Gastronomy & Visual Plating
> **SE-B | i22582, i221513**  
> A Hybrid-SOTA Pipeline for Neural Recipe Generation & Conditional Image Synthesis.

---

## 🧠 System Architecture: The Triple-Model Pipeline
PlatePal has transitioned from a baseline DCGAN/GPT-2 setup to a professional-grade GenAI architecture designed for high-fidelity culinary outputs.

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Knowledge (RAG)** | **FAISS** + `all-MiniLM-L6-v2` | Semantic search over 20k recipes to ground AI in real data. |
| **Intelligence (LLM)** | **Llama-3.3-70B** (via Groq) | Context-aware reasoning and technical recipe synthesis. |
| **Aesthetics (Diffusion)** | **Stable Diffusion v1.5** + **LoRA** | Photorealistic food plating visuals with regional fine-tuning. |

---

## 📂 Project Structure
```text
PlatePal/
├── app/                          # Core backend (FastAPI)
│   ├── main.py                   # API Orchestration
│   ├── models/
│   │   ├── text_gen_groq.py      # Llama-3.3 / Groq Integration
│   │   └── image_gen.py          # SD v1.5 + Custom LoRA
│   └── utils/
│       ├── recipe_db.py          # Semantic RAG Engine (FAISS)
│       └── data_preprocessing.py # Dataset ETL (RecipeNLG / CAFD)
├── web/                          # Modern Frontend (React + Vite)
├── scripts/                      # Indexing & Testing utilities
│   ├── build_rag_index.py        # Vector indexing for 20k recipes
│   └── test_rag.py               # RAG retrieval validation
├── VIVA_PREP_GUIDE.md            # Technical & Mathematical deep-dive
├── PROJECT_MODULE_DIVISION.md    # Group partner role breakdown
└── requirements.txt              # Standardized GenAI stack
```

---

## 🚀 The Technical Workflow
1.  **User Input**: Receives raw ingredients (unstructured text).
2.  **Retrieval**: The **Semantic RAG** engine converts input into a vector and searches the **FAISS** index for the 3 most relevant professional recipes.
3.  **Augmentation**: These recipes are injected into the **Llama-3.3** system prompt as technical context.
4.  **Generation (Text)**: The LLM synthesizes a structured recipe, constrained by low-temperature sampling ($\tau = 0.3$) for accuracy.
5.  **Generation (Image)**: The dish title is passed to the **Stable Diffusion** pipeline.
6.  **Style Injection**: A custom **LoRA (Low-Rank Adaptation)** layer activates, applying specific "South Asian" plating textures learned from the CAFD dataset.

---

## 🛠️ Advanced GenAI Concepts Applied
*   **RAG (Retrieval-Augmented Generation)**: Mitigates LLM hallucinations by providing real-world culinary "ground truth."
*   **PEFT (Parameter-Efficient Fine-Tuning)**: Uses LoRA to adapt a 800M+ parameter model using only 16MB adapters.
*   **Classifier-Free Guidance (CFG)**: Steers the diffusion process to match the recipe prompt while filtering for high-end aesthetics.
*   **Semantic Search**: Uses L2-distance in high-dimensional vector space to find matches based on *meaning* rather than exact keywords.

---

## 📦 Setup & Installation
1.  **Environment**: Create a virtual env and run `pip install -r requirements.txt`.
2.  **Keys**: Copy `.env.example` to `.env` and add your `GROQ_API_KEY`.
3.  **Index**: Run `python scripts/build_rag_index.py` to prepare the RAG memory.
4.  **Run**:
    ```bash
    # Terminal 1: Backend
    uvicorn app.main:app --reload
    
    # Terminal 2: Frontend
    cd web && npm run dev
    ```

---

## 📊 Evaluation & Metrics
*   **Text**: Perplexity (fluency), BLEU-4 (overlap), and Ingredient Coverage.
*   **Image**: **FID (Fréchet Inception Distance)** to measure the similarity between generated plates and real professional food photography.

---
**PlatePal v3.0 | Optimized for RTX 5060 Ti | GenAI Assignment #3**
