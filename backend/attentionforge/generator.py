"""
Seed-aware generation helper for AttentionForge Studio.

This project is an interpretability studio, not a general chatbot.
The default coherent mode generates a clean explanation around the user's seed
without pretending that the raw PyTorch mini Transformer has world knowledge.

Raw Transformer sampling remains available separately.
"""

from __future__ import annotations

import re


STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "about",
    "what", "when", "where", "which", "your", "you", "are", "will", "can",
    "how", "why", "does", "make", "made", "using", "use", "give", "tell",
}


def extract_focus(seed_text: str) -> str:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*", seed_text.lower())
    meaningful = [token for token in tokens if token not in STOPWORDS]

    if meaningful:
        return " ".join(meaningful[:4])

    clean = seed_text.strip()
    return clean if clean else "the selected input"


def classify_seed(seed_text: str) -> str:
    text = seed_text.lower()

    if any(word in text for word in ["attention", "head", "query", "key", "value", "token"]):
        return "attention"

    if any(word in text for word in ["mask", "causal", "future", "decoder"]):
        return "masking"

    if any(word in text for word in ["position", "positional", "encoding", "order", "sequence"]):
        return "position"

    if any(word in text for word in ["transformer", "layer", "block", "model", "neural"]):
        return "architecture"

    return "custom"


def coherent_generate(seed_text: str, max_new_tokens: int = 120) -> str:
    focus = extract_focus(seed_text)
    topic = classify_seed(seed_text)

    if topic == "attention":
        sentences = [
            f'The phrase "{focus}" is analyzed as a token sequence so the studio can show how attention routes information between tokens.',
            "Each row in the heatmap acts like a query, and each column acts like a possible source of context.",
            "The strongest links reveal which earlier tokens are being emphasized by the selected layer and head.",
        ]

    elif topic == "masking":
        sentences = [
            f'The phrase "{focus}" is useful for demonstrating how causal masking controls information flow.',
            "When masking is enabled, each token can only attend to itself and earlier tokens.",
            "This prevents future-token leakage and makes the setup consistent with decoder-style next-token prediction.",
        ]

    elif topic == "position":
        sentences = [
            f'The phrase "{focus}" is used to demonstrate why Transformers need positional information.',
            "Self-attention compares tokens in parallel, so positional encoding gives the model a separate signal for order.",
            "The encoding view shows how positions receive different numeric patterns before attention is calculated.",
        ]

    elif topic == "architecture":
        sentences = [
            f'The phrase "{focus}" is passed through a larger educational Transformer preset with multiple layers and heads.',
            "Each block combines attention, residual connections, layer normalization, and a feed-forward network.",
            "The goal is to make the architecture inspectable rather than hidden behind a high-level API.",
        ]

    else:
        sentences = [
            f'The seed "{focus}" is treated as the focus input for this interpretability demo.',
            "AttentionForge Studio does not claim to be a general chatbot; it shows how input tokens are routed through Transformer-style attention.",
            "Use the workbench to inspect how the selected layer and head assign attention weights across the token sequence.",
        ]

    # Respect max_new_tokens without repeating sentences.
    words = []
    for sentence in sentences:
        for word in sentence.split():
            if len(words) >= max_new_tokens:
                break
            words.append(word)
        if len(words) >= max_new_tokens:
            break

    output = " ".join(words).strip()

    # Try to end cleanly at a sentence boundary.
    last_end = max(output.rfind("."), output.rfind("?"), output.rfind("!"))
    if last_end > 35:
        output = output[: last_end + 1]

    if not output.endswith((".", "?", "!")):
        output += "."

    if seed_text.strip():
        return f"{seed_text.strip()} — {output}"

    return output
