"""
PlatePal — Evaluation Metrics
==============================

Text Metrics:
  - Perplexity     (confidence / fluency)
  - BLEU-4         (n-gram overlap)
  - ROUGE-L        (sequential logic)

Image Metrics:
  - FID            (Fréchet Inception Distance)
"""

import math
import torch
import numpy as np
from typing import List, Dict
from collections import Counter


# ─────────────────────────────────────────────────────────────────────
# TEXT METRICS
# ─────────────────────────────────────────────────────────────────────

def compute_perplexity(model, tokenizer, texts: List[str], device: str = "cpu", max_length: int = 512) -> float:
    """
    Compute average perplexity of the model over a list of texts.
    Lower perplexity = more confident / fluent model.
    """
    model.eval()
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for text in texts:
            enc = tokenizer(text, return_tensors="pt", max_length=max_length,
                            truncation=True, padding=True).to(device)
            labels = enc["input_ids"].clone()
            labels[labels == tokenizer.pad_token_id] = -100

            outputs = model(input_ids=enc["input_ids"],
                            attention_mask=enc["attention_mask"],
                            labels=labels)
            total_loss += outputs.loss.item()
            count += 1

    avg_loss = total_loss / max(count, 1)
    return math.exp(min(avg_loss, 100))


def _ngrams(tokens: List[str], n: int) -> Counter:
    """Extract n-gram counts from token list."""
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def compute_bleu4(reference: str, hypothesis: str) -> float:
    """
    Compute BLEU-4 score between a reference and hypothesis string.
    """
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    if len(hyp_tokens) == 0:
        return 0.0

    # Brevity penalty
    bp = min(1.0, math.exp(1 - len(ref_tokens) / max(len(hyp_tokens), 1)))

    # Precision for n=1..4
    precisions = []
    for n in range(1, 5):
        ref_ngrams = _ngrams(ref_tokens, n)
        hyp_ngrams = _ngrams(hyp_tokens, n)

        clipped = sum(min(hyp_ngrams[ng], ref_ngrams[ng]) for ng in hyp_ngrams)
        total = max(sum(hyp_ngrams.values()), 1)
        precisions.append(clipped / total)

    # Geometric mean (with smoothing for zero precisions)
    log_avg = sum(math.log(max(p, 1e-10)) for p in precisions) / 4
    return bp * math.exp(log_avg)


def compute_rouge_l(reference: str, hypothesis: str) -> float:
    """
    Compute ROUGE-L F1 score using Longest Common Subsequence.
    """
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    m, n = len(ref_tokens), len(hyp_tokens)
    if m == 0 or n == 0:
        return 0.0

    # LCS via dynamic programming
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i-1] == hyp_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    lcs_len = dp[m][n]
    precision = lcs_len / n
    recall = lcs_len / m
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate_text_quality(
    model, tokenizer, generated_recipes: List[str],
    reference_recipes: List[str] = None, device: str = "cpu"
) -> Dict[str, float]:
    """
    Run the full text evaluation suite.
    """
    results = {}

    # Perplexity
    ppl = compute_perplexity(model, tokenizer, generated_recipes, device)
    results["perplexity"] = ppl

    # BLEU-4 and ROUGE-L (require references)
    if reference_recipes and len(reference_recipes) == len(generated_recipes):
        bleu_scores = []
        rouge_scores = []
        for ref, hyp in zip(reference_recipes, generated_recipes):
            bleu_scores.append(compute_bleu4(ref, hyp))
            rouge_scores.append(compute_rouge_l(ref, hyp))
        results["bleu4"] = np.mean(bleu_scores)
        results["rouge_l"] = np.mean(rouge_scores)

    return results


# ─────────────────────────────────────────────────────────────────────
# IMAGE METRICS — FID (Fréchet Inception Distance)
# ─────────────────────────────────────────────────────────────────────

def get_inception_features(images: torch.Tensor, device: str = "cpu", batch_size: int = 64) -> np.ndarray:
    """
    Extract Inception-v3 features from a batch of images.
    Images should be (N, 3, H, W) tensors in [0, 1] range.
    They will be resized to 299x299 internally.
    """
    from torchvision.models import inception_v3, Inception_V3_Weights
    import torch.nn.functional as F

    model = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False).to(device)
    model.eval()

    # Remove the final classification layer → get 2048-dim features
    model.fc = torch.nn.Identity()

    all_features = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size].to(device)
        # Resize to 299x299 for Inception
        batch = F.interpolate(batch, size=(299, 299), mode="bilinear", align_corners=False)
        with torch.no_grad():
            feats = model(batch)
        all_features.append(feats.cpu().numpy())

    return np.concatenate(all_features, axis=0)


def compute_fid(real_features: np.ndarray, fake_features: np.ndarray) -> float:
    """
    Compute Fréchet Inception Distance between two sets of Inception features.

    FID = ||mu_r - mu_f||^2 + Tr(Sigma_r + Sigma_f - 2*(Sigma_r @ Sigma_f)^{1/2})
    """
    from scipy.linalg import sqrtm

    mu_r = np.mean(real_features, axis=0)
    mu_f = np.mean(fake_features, axis=0)
    sigma_r = np.cov(real_features, rowvar=False)
    sigma_f = np.cov(fake_features, rowvar=False)

    # Regularize for numerical stability (prevents NaN with small batch sizes)
    eps = 1e-6
    sigma_r += eps * np.eye(sigma_r.shape[0])
    sigma_f += eps * np.eye(sigma_f.shape[0])

    diff = mu_r - mu_f
    covmean, _ = sqrtm(sigma_r @ sigma_f, disp=False)

    # Handle numerical instability
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma_r + sigma_f - 2 * covmean)
    return float(fid)


def evaluate_image_quality(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    device: str = "cpu"
) -> Dict[str, float]:
    """
    Run image evaluation: compute FID between real and generated images.
    Both inputs should be (N, 3, H, W) tensors in [0, 1] range.
    """
    print("[Eval] Extracting Inception features from real images...")
    real_feats = get_inception_features(real_images, device)

    print("[Eval] Extracting Inception features from fake images...")
    fake_feats = get_inception_features(fake_images, device)

    fid = compute_fid(real_feats, fake_feats)
    print(f"[Eval] FID: {fid:.2f}")

    return {"fid": fid}


# ─────────────────────────────────────────────────────────────────────
# Ingredient Coverage Check
# ─────────────────────────────────────────────────────────────────────

def ingredient_coverage(ingredients: str, recipe_text: str) -> float:
    """
    Compute the fraction of input ingredients that appear in the generated recipe.
    A coverage < 0.8 indicates the model needs more fine-tuning.
    """
    ing_list = [i.strip().lower() for i in ingredients.split(",")]
    recipe_lower = recipe_text.lower()
    mentioned = sum(1 for ing in ing_list if ing in recipe_lower)
    return mentioned / max(len(ing_list), 1)


# ─────────────────────────────────────────────────────────────────────
# Full Evaluation Report
# ─────────────────────────────────────────────────────────────────────

def generate_report(text_results: Dict, image_results: Dict = None) -> str:
    """Generate a human-readable evaluation report."""
    lines = [
        "=" * 50,
        " PlatePal — Evaluation Report",
        "=" * 50,
        "",
        "TEXT GENERATION METRICS",
        "-" * 30,
        f"  Perplexity:  {text_results.get('perplexity', 'N/A'):.2f}",
    ]
    if "bleu4" in text_results:
        lines.append(f"  BLEU-4:      {text_results['bleu4']:.4f}")
    if "rouge_l" in text_results:
        lines.append(f"  ROUGE-L:     {text_results['rouge_l']:.4f}")

    if image_results:
        lines += [
            "",
            "IMAGE SYNTHESIS METRICS",
            "-" * 30,
            f"  FID:         {image_results.get('fid', 'N/A'):.2f}",
        ]

    lines.append("\n" + "=" * 50)
    return "\n".join(lines)


if __name__ == "__main__":
    # Quick self-test with dummy data
    print("=== Text Metrics Self-Test ===")
    ref = "Preheat the oven to 350 degrees. Mix flour and sugar. Add eggs and butter."
    hyp = "Preheat the oven to 350 degrees. Combine flour with sugar. Add the eggs."
    print(f"BLEU-4:  {compute_bleu4(ref, hyp):.4f}")
    print(f"ROUGE-L: {compute_rouge_l(ref, hyp):.4f}")

    cov = ingredient_coverage("chicken, rice, garlic", "Cook the chicken with garlic and serve over rice.")
    print(f"Coverage: {cov:.0%}")
