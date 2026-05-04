# PlatePal: Project Module Division (Viva Reference)

To ensure a fair and technically rigorous assessment, the PlatePal codebase is divided into two primary roles. 

## Partner 1: The GenAI & Retrieval Engineer
*Focus: Algorithms, Mathematical Foundations, and Model Adaptation.*

### Module 1: Semantic RAG & Vector Indexing
- **Core Logic**: Implementing the `SemanticRecipeRetriever`.
- **Tech Stack**: FAISS (Facebook AI Similarity Search), SentenceTransformers (`all-MiniLM-L6-v2`).
- **Mathematical Responsibility**: Explaining L2 (Euclidean) distance for vector similarity search.
- **Key File**: `app/utils/recipe_db.py`, `scripts/build_rag_index.py`

### Module 2: NLP Synthesis (Llama-3.3 Pipeline)
- **Core Logic**: Managing the high-fidelity text generation via Groq.
- **Tech Stack**: Groq API, Llama-3.3-70B, GPT-2 (Fallback).
- **Responsibility**: Instruction engineering, context injection (RAG -> LLM), and temperature control.
- **Key File**: `app/models/text_gen_groq.py`

### Module 3: Visual Synthesis (Diffusion & LoRA)
- **Core Logic**: Implementing the Stable Diffusion v1.5 pipeline with custom LoRA weights.
- **Tech Stack**: PyTorch, Diffusers, PEFT (LoRA).
- **Mathematical Responsibility**: Explaining Low-Rank Adaptation ($W = W_0 + BA$) and Classifier-Free Guidance (CFG).
- **Key File**: `app/models/image_gen.py`, `train_sd_lora.py`

---

## Partner 2: The Full-Stack & Data Architect
*Focus: Integration, UX/UI, and Empirical Validation.*

### Module 4: API Orchestration & System Integration
- **Core Logic**: The FastAPI backend coordinating the multi-modal request flow.
- **Tech Stack**: FastAPI, Pydantic, Uvicorn.
- **Responsibility**: End-to-end request lifecycle, Base64 image serialization, and backend/frontend handshake.
- **Key File**: `app/main.py`

### Module 5: Premium User Experience (UX/UI)
- **Core Logic**: Developing the high-end React frontend.
- **Tech Stack**: React, Vite, CSS (Glassmorphism / Typography).
- **Responsibility**: Developing the "Michelin-Star" aesthetic and the complex parsing logic for structured AI responses.
- **Key File**: `web/src/App.jsx`

### Module 6: Data Engineering & Preprocessing
- **Core Logic**: Cleaning and filtering the RecipeNLG and Food-101 datasets.
- **Tech Stack**: Pandas, JSON, Regex.
- **Responsibility**: ETL pipeline, filtering "trivial" data, and preparing datasets for fine-tuning.
- **Key File**: `app/utils/data_preprocessing.py`

### Module 7: Evaluation & Metrics
- **Core Logic**: Quantitative performance measurement.
- **Tech Stack**: FID Scores, BLEU, ROUGE, Perplexity.
- **Responsibility**: Proving model improvement over baselines via standardized AI metrics.
- **Key File**: `run_evaluation.py`
