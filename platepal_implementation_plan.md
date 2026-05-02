# Neural Architectural Framework and Implementation Roadmap for PlatePal

## 1. Executive Summary
PlatePal is an AI-powered Recipe and Plating Visual Generator designed to solve everyday food decision fatigue and household food waste. By taking a discrete set of input ingredients from the user, the application leverages Generative AI to output a complete, tailored step-by-step recipe alongside a photorealistic image of the finished, professionally plated dish.

This implementation plan details a multi-stage approach that harmonizes Natural Language Processing (NLP) for coherent recipe generation and Computer Vision (CV) for conditional image synthesis. The roadmap is optimized for a rapid 48-hour development cycle, targeting execution on a 16GB RTX 5060 Ti.

## 2. Neural Architectural Framework
The core objective of the architecture is to map a discrete set of input ingredients into:
1. A continuous semantic space for structured text generation.
2. A high-dimensional pixel space for conditional image synthesis.

The system relies on a Text-to-Text Transformer architecture for the recipe text and a Deep Convolutional Generative Adversarial Network (DCGAN) for the image visualization, bridged by shared latent representations using a pre-trained CLIP model.

## 3. Detailed Dataset Specifications
Due to the strict development deadline, the project will utilize high-quality subsets of established datasets.

| Dataset | Modality | Purpose | Details |
| :--- | :--- | :--- | :--- |
| **RecipeNLG** (Cleaned Subset) | Text | Fine-tuning the Transformer for structured recipe steps. | Focuses on structure and logic. Preprocessing will filter for "rich" recipes, removing those with vague phrases (e.g., "mix all") or very short instructions to prevent trivial outputs. Key fields: `title`, `ingredients`, `directions`, and `NER` (food entities). |
| **Food-101** (Subset) | Image | Training the base DCGAN/CVAE. | Utilizing `food_c101_n10099_r64x64x3.h5`, containing ~11,000 pre-processed 64x64 color images across 101 categories. |
| **CAFD** (Central Asian Food Dataset) | Image | Regional plating and cultural specificity. | Contains 16,499 images across 42 classes. Emphasizes South/Central Asian regional novelty (e.g., Biryani, Nihari, Karahi) for specific authenticity. |

## 4. Development Roadmap (48-Hour Execution)

### Step 1: Text Generation Implementation
*   **Architecture:** Pre-trained DistilGPT-2 or GPT-2 (124M) as the base foundation model.
*   **Tokenization:** Recipes must be prepended with special control tokens to enforce rigid structural generation:
    *   `<RECIPE_START>`
    *   `<INPUT_START>` (to encapsulate ingredient entities)
    *   `<INSTR_START>` (to signify the beginning of instructions)
*   **Training Strategy:** 
    *   Objective: Autoregressive next-token prediction.
    *   Data: Fine-tune on a subset of 10k–20k highly structured recipes.
    *   Hyperparameters: Learning rate of $2 \times 10^{-4}$ utilizing a warm-up schedule to ensure stability during early epochs.

### Step 2: Image Synthesis Implementation (DCGAN)
*   **Generator:** 
    *   Input: A 100-dimensional noise vector ($z$) concatenated with a text embedding.
    *   Architecture: Four `ConvTranspose2d` layers paired with `BatchNorm` and `ReLU` activations.
    *   Output: 64x64x3 photorealistic image.
*   **Discriminator:** 
    *   Architecture: A binary classifier utilizing four `Conv2d` layers with `LeakyReLU` and `BatchNorm`.
    *   Objective: Distinguish between real dataset images and fake generator outputs.
*   **Conditioning (The Bridge):** 
    *   A pre-trained CLIP model will extract fixed 512-dimensional embeddings derived from the generated recipe title and input ingredients.
    *   This embedding acts as a conditional vector injected directly into the Generator to ensure the output image matches the text semantics.

## 5. Model Training Quality Checklist
During training, strictly adhere to this checklist to identify issues early and avoid wasting compute cycles:

- [ ] **Loss Stability:** Monitor GAN training dynamics. The Discriminator loss should not drop to zero immediately (indicating an overpowered critic). Ensure Generator loss decreases as it learns to "fool" the critic.
- [ ] **Real-Time Visuals:** Save sample reconstructions (e.g., `sample_epoch_5.png`) every 5 epochs. If outputs are visually identical across different random samples, Mode Collapse has occurred.
- [ ] **Gradient Health:** Check for vanishing gradients (especially if reverting to deep CNNs or LSTMs). Implement Spectral Normalization or ensure BatchNorm layers are functioning to maintain gradient flow.
- [ ] **Logic Check (Text):** Evaluate the first 10 generated recipes. If the model repeats instructions or hallucinates ingredients absent from the prompt, increase the dropout rate or adjust the teacher forcing ratio.
- [ ] **Ingredient Coverage:** Validate that generated steps actually utilize the provided ingredients. A coverage ratio $< 0.8$ strongly indicates the model requires further fine-tuning.

## 6. Hardware and Workload Optimization
Optimized for a local GPU setup (NVIDIA RTX 5060 Ti 16GB):
*   **GPU Utilization:** Set a batch size of `128` for the DCGAN and `32` for the Transformer to maximize VRAM throughput without causing Out-of-Memory (OOM) errors.
*   **Mixed Precision:** Enable `torch.cuda.amp` to accelerate training. Utilizing FP16 will drastically speed up matrix multiplications while preserving necessary accuracy.
*   **Dataset Bottleneck Prevention:** Load images from pre-scaled 64x64 HDF5 format (`.h5`). Avoid on-the-fly resizing and CPU-bound augmentations, which will throttle epoch speeds.

## 7. UI/Frontend and Integration
*   **Backend Interface:** Host both the text and image models via **FastAPI**.
    *   Endpoint Flow: Receive ingredients $\rightarrow$ Trigger GPT text model $\rightarrow$ Pass generated text to CLIP/DCGAN image model $\rightarrow$ Return JSON payload with text and image bytes.
*   **Frontend Interface:** Build a single-page web application utilizing **Streamlit**.
    *   The UI must feature a clean, side-by-side display showcasing the structured recipe card next to the generated plating visual.

## 8. Evaluation Strategy
To rigorously validate the generative capabilities, use the following metrics:
*   **Text Metrics:**
    *   **Perplexity:** To gauge model confidence and fluency.
    *   **BLEU-4:** To measure n-gram overlap and syntactical similarity against ground-truth recipes.
    *   **ROUGE-L:** To evaluate the sequential logic and longest common subsequences of the instructions.
*   **Image Metrics:**
    *   **Fréchet Inception Distance (FID):** To quantify the visual quality and measure the distribution similarity between the generated plating images and the real Food-101/CAFD dataset images. Lower FID indicates higher photorealism.
