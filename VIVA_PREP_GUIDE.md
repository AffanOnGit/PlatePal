# PlatePal: High-Fidelity GenAI VIVA Preparation Guide
> **Subject**: Generative AI (SE-B)  
> **Authors**: i22582, i221513  
> **Topic**: Neural Recipe Generation & Conditional Image Synthesis

---

## 1. The Executive Problem-Solution Framework
**The Problem**: Food decision fatigue and household food waste.  
**The Solution**: An AI-powered personal sous-chef that bridges the gap between raw ingredients (unstructured data) and professional culinary outputs (structured text + high-res visuals).

**Technical Achievement**: A Hybrid-SOTA pipeline combining **Cloud-LLMs (Llama-3.3)**, **Vector-Retrieval (RAG)**, and **Parameter-Efficient Fine-Tuning (LoRA)**.

---

## 2. Project Evolution (The "Architecture History")
*Evaluators love to see that you dealt with problems. Use this history to show growth.*

| Phase | NLP Engine | Image Engine | Problem Encountered | Solution Applied |
| :--- | :--- | :--- | :--- | :--- |
| **v1 (Proposal)** | GPT-2 (Local) | DCGAN (64x64) | Model Hallucinations & Blurry Images | Upgrade to SD & Llama 3 |
| **v2 (Migration)** | Llama-3.3 (Groq) | Stable Diffusion v1.5 | General food visuals (Base SD lacks "South Asian" plating) | **LoRA Fine-tuning** |
| **v3 (Final)** | Llama-3.3 + **RAG** | SD v1.5 + **Custom LoRA** | AI was "guessing" ingredients logic | **Semantic RAG (FAISS)** |

---

## 3. Deep Dive: Semantic RAG (Retrieval-Augmented Generation)
**Files Involved**: `app/utils/recipe_db.py`, `scripts/build_rag_index.py`

### Why RAG?
Llama-3.3 is a 70B parameter giant, but it doesn't "know" your specific local pantry logic or exact verified recipes. RAG provides the **"Ground Truth"** to prevent hallucinations.

### The Inner Workings:
1.  **Indexing**: We took 20,000 recipes from the `RecipeNLG` dataset.
2.  **Vectorization**: Using `all-MiniLM-L6-v2`, we mapped each recipe to a **384-dimensional dense vector**.
3.  **Retrieval Logic**:
    *   **The Math (Euclidean Distance)**: We use FAISS to calculate the $L2$ distance between the user's query ($q$) and stored recipe vectors ($v$):
        $$d(q, v) = \sqrt{\sum_{i=1}^{n} (q_i - v_i)^2}$$
    *   **Parameter `top_k=3`**: Why 3? 
        *   **Redundancy**: If the first result is a slight outlier, the other two provide "majority consensus."
        *   **Context Window**: Llama-3.3 has a large window, but injecting 10 recipes causes "Lost in the Middle" bias. 3 is the sweet spot for professional grounding.

---

## 4. Deep Dive: Llama-3.3 & Instruction Tuning
**Files Involved**: `app/models/text_gen_groq.py`, `app/main.py`

### The Communication Flow:
1.  **Frontend**: Receives ingredients.
2.  **RAG**: Finds 3 similar recipes.
3.  **Prompt Construction**: 
    ```python
    system_prompt = "You are a Michelin-star Executive Chef. Use the FOLLOWING 3 REAL RECIPES as context..."
    ```
4.  **Temperature Logic**: We set `Temperature = 0.3`.
    *   **Reasoning**: High temperature ($\tau = 0.9$) makes the AI "creative"—it might suggest cooking rice for 2 minutes. Low temperature ($\tau = 0.3$) makes it **deterministic** and **technically accurate**.

---

## 5. Deep Dive: Stable Diffusion & LoRA
**Files Involved**: `app/models/image_gen.py`, `train_sd_lora.py`

### The Diffusion Math
Stable Diffusion works by reversing a Markov Chain of noise.
*   **Loss Function**:
    $$L_{ldm} := \mathbb{E}_{x, \epsilon \sim \mathcal{N}(0,1), t} [\|\epsilon - \epsilon_\theta(z_t, t, \tau_\theta(y))\|_2^2]$$
    It tries to predict the noise $\epsilon$ added to the latent $z$ based on text conditioning $y$.

### LoRA (Low-Rank Adaptation)
Why did we use LoRA instead of full fine-tuning?
*   **Efficiency**: We only train **Rank-16 matrices**.
*   **The Equation**:
    $$W_{updated} = W_{frozen} + (B \times A)$$
    Where $B$ and $A$ are small matrices. This allows the model to learn "South Asian Plating" without losing its general knowledge of "what a plate is."

### Image Parameters:
*   **CFG Scale (8.0)**: **Classifier-Free Guidance**. Higher values (8.0+) force the image to match the text prompt more strictly.
*   **Steps (30)**: We use the **DDPMScheduler**. 30 steps is the convergence point where quality plateaus and speed is maximized.

---

## 6. End-to-End Workflow (Evaluation Day Trace)

1.  **User Input**: `{"ingredients": "chicken, rice, yogurt"}`
2.  **Semantic Search**: `recipe_db.py` finds "Hyderabadi Biryani" as a Top-1 match in the 20k dataset.
3.  **LLM Call**: Llama-3.3 receives the user prompt + the Biryani technical context.
4.  **Text Output**: AI generates a structured recipe with `<TITLE_START>Royal Biryani<INSTR_START>...`.
5.  **Image Trigger**: The title "Royal Biryani" is passed to **Stable Diffusion**.
6.  **The LoRA layer** activates, adding specific textures (saffron yellow, ghee-glistening rice) learned from the CAFD dataset.
7.  **Final Polish**: Negative prompts like `"blurry, plastic"` are applied to ensure the output is appetizing.

---

## 7. Key Terminologies for the VIVA
*   **Hallucination**: When the LLM generates plausible-sounding but factually wrong cooking steps. (Solved by RAG).
*   **Mode Collapse**: When a GAN (like our v1 baseline) only generates one type of image. (Solved by switching to Diffusion).
*   **Zero-Shot Prompting**: Asking the model to do a task without examples.
*   **Few-Shot Prompting**: Providing the RAG recipes as "examples" for the LLM to follow.
*   **Embeddings**: The numerical representation of "meaning" in text.

---

## 8. Summary of Parameters
| Variable | Value | Reasoning |
| :--- | :--- | :--- |
| `top_k` (RAG) | 3 | Diversity vs. Context limit balance. |
| `Temperature` | 0.3 | High precision for recipe accuracy. |
| `LoRA Rank (r)`| 16 | Learning specific textures without "catastrophic forgetting." |
| `Batch Size` | 2 | Constraints of a 16GB GPU during LoRA training. |
| `CFG Scale` | 8.0 | Strong adherence to the "Michelin-Star" aesthetic. |

---
**Prepared for Rigorous Assessment | PlatePal v3.0**
