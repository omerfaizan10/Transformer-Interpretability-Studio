"""
FastAPI backend for AttentionForge Studio.

Run:
    python -m uvicorn backend.main:app --reload --port 8000
"""

from __future__ import annotations

from typing import Any
from pathlib import Path
import re

import torch
import torch.nn.functional as F
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.attentionforge.generator import coherent_generate
from backend.attentionforge.claude_client import (
    claude_is_configured,
    explain_attention_with_claude,
    generate_with_claude,
    get_claude_model,
    get_env_debug,
)
from backend.attentionforge.model import MiniTransformerLM
from backend.attentionforge.tokenizer import DEFAULT_CORPUS, WordTokenizer


class AnalyzeRequest(BaseModel):
    text: str = Field(default="attention lets each token decide which earlier tokens matter", max_length=320)
    layer: int = Field(default=0, ge=0)
    head: int = Field(default=0, ge=0)
    causal_mask: bool = True
    analysis_mode: str = Field(default="guided")


class GenerateRequest(BaseModel):
    seed_text: str = Field(default="attention", max_length=160)
    max_new_tokens: int = Field(default=120, ge=20, le=260)
    temperature: float = Field(default=0.8, ge=0.1, le=2.0)
    mode: str = Field(default="claude")


class ClaudeExplainRequest(BaseModel):
    input_text: str = Field(default="attention lets each token decide which earlier tokens matter", max_length=320)
    layer: int = Field(default=0, ge=0)
    head: int = Field(default=0, ge=0)
    causal_mask: bool = True
    analysis_mode: str = Field(default="guided")


torch.manual_seed(42)
torch.set_num_threads(2)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = PROJECT_ROOT / "backend" / "checkpoints" / "raw_transformer_trained.pt"

tokenizer = WordTokenizer.from_text(DEFAULT_CORPUS)
TRAINED_CONFIG = {
    "vocab_size": tokenizer.vocab_size,
    "d_model": 64,
    "num_heads": 4,
    "num_layers": 3,
    "ff_hidden_dim": 256,
    "max_len": 96,
    "dropout": 0.0,
}

model = MiniTransformerLM(**TRAINED_CONFIG)
CHECKPOINT_LOADED = False
CHECKPOINT_INFO = {}

if CHECKPOINT_PATH.exists():
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        CHECKPOINT_LOADED = True
        CHECKPOINT_INFO = checkpoint.get("training", {})
    except Exception as exc:
        CHECKPOINT_INFO = {"error": str(exc)}

model.eval()

app = FastAPI(
    title="AttentionForge Studio API",
    description="Professional raw PyTorch Transformer interpretability backend.",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def prepare_input(text: str) -> tuple[torch.Tensor, list[str]]:
    ids = tokenizer.safe_encode(text)
    display_tokens = tokenizer.display_tokens(text)

    if not display_tokens:
        display_tokens = tokenizer.display_tokens("attention")

    ids = ids[-model.max_len :]
    display_tokens = display_tokens[-model.max_len :]

    idx = torch.tensor([ids], dtype=torch.long)
    return idx, display_tokens


def matrix_to_list(tensor: torch.Tensor, decimals: int = 5) -> list[list[float]]:
    matrix = tensor.detach().cpu()
    matrix = torch.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    rounded = torch.round(matrix * (10**decimals)) / (10**decimals)
    return rounded.tolist()


def guided_attention_matrix(
    tokens: list[str],
    layer: int,
    head: int,
    causal_mask: bool,
) -> torch.Tensor:
    """
    Creates an interpretable attention matrix for presentation mode.

    Raw mode still returns the actual untrained PyTorch attention weights.
    Guided mode prevents the demo from looking random while teaching the same concepts:
    self attention, local attention, first-token anchoring, repeated-token links, and semantic grouping.
    """
    n = len(tokens)
    if n == 0:
        return torch.zeros(0, 0)

    topic_groups = [
        {"attention", "attends", "attend", "head", "heads", "query", "queries", "key", "keys", "value", "values"},
        {"token", "tokens", "word", "words", "text", "input"},
        {"mask", "masked", "masking", "causal", "future", "decoder"},
        {"position", "positional", "encoding", "order", "sequence"},
        {"transformer", "model", "layer", "layers", "block", "blocks"},
    ]

    matrix = torch.zeros(n, n)

    for i, query in enumerate(tokens):
        allowed = [j for j in range(n) if (not causal_mask or j <= i)]

        if not allowed:
            allowed = [i]

        scores = torch.zeros(n)

        for j in allowed:
            key = tokens[j]
            score = 0.08

            # Head behavior changes slightly so heads do not all look identical.
            if head % 6 == 0:
                if i == j:
                    score += 1.25
                if j == max(0, i - 1):
                    score += 0.70

            elif head % 6 == 1:
                if j == 0:
                    score += 0.95
                if i == j:
                    score += 0.45

            elif head % 6 == 2:
                if query == key:
                    score += 1.10
                if i == j:
                    score += 0.45

            elif head % 6 == 3:
                if abs(i - j) <= 2:
                    score += 0.85
                if i == j:
                    score += 0.35

            elif head % 6 == 4:
                for group in topic_groups:
                    if query in group and key in group:
                        score += 1.05
                if i == j:
                    score += 0.25

            else:
                if j == 0:
                    score += 0.55
                if j == max(0, i - 1):
                    score += 0.55
                if query == key:
                    score += 0.55

            # Deeper layers become more global.
            if layer >= 3 and j == 0:
                score += 0.20
            if layer >= 4 and abs(i - j) <= 3:
                score += 0.18

            scores[j] = score

        # Block future positions exactly in guided mode too.
        if causal_mask:
            for j in range(i + 1, n):
                scores[j] = 0.0

        row_sum = scores.sum()
        if row_sum <= 0:
            scores[i] = 1.0
            row_sum = 1.0

        matrix[i] = scores / row_sum

    return matrix


def top_attention_links(attention: torch.Tensor, tokens: list[str], limit: int = 10) -> list[dict[str, Any]]:
    attn = attention.detach().cpu()
    links = []
    seq_len = attn.shape[0]

    for query_idx in range(seq_len):
        for key_idx in range(seq_len):
            weight = float(attn[query_idx, key_idx])
            if weight <= 0:
                continue
            links.append(
                {
                    "query_index": query_idx,
                    "query_token": tokens[query_idx],
                    "key_index": key_idx,
                    "key_token": tokens[key_idx],
                    "weight": round(weight, 5),
                    "is_self_attention": query_idx == key_idx,
                }
            )

    links.sort(key=lambda item: item["weight"], reverse=True)
    return links[:limit]


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "product": "AttentionForge Studio",
        "author": "Omer Faizan",
        "claude_configured": claude_is_configured(),
        "claude_model": get_claude_model(),
        "checkpoint_loaded": CHECKPOINT_LOADED,
    }


@app.get("/api/config")
def config():
    total_params = sum(p.numel() for p in model.parameters())
    return {
        "product": "AttentionForge Studio",
        "author": "Omer Faizan",
        "model_name": "MiniTransformerLM",
        "vocab_size": tokenizer.vocab_size,
        "vocabulary": tokenizer.vocab,
        "d_model": model.d_model,
        "num_heads": model.num_heads,
        "num_layers": model.num_layers,
        "ff_hidden_dim": model.ff_hidden_dim,
        "max_len": model.max_len,
        "parameter_count": total_params,
        "checkpoint_loaded": CHECKPOINT_LOADED,
        "checkpoint_path": str(CHECKPOINT_PATH),
        "checkpoint_info": CHECKPOINT_INFO,
        "generation_modes": ["claude", "coherent", "raw"],
        "claude_configured": claude_is_configured(),
        "claude_model": get_claude_model(),
        "analysis_modes": ["guided", "raw"],
        "implementation": "Raw PyTorch. No Hugging Face. No nn.Transformer. No nn.MultiheadAttention.",
    }


@app.post("/api/analyze")
@torch.no_grad()
def analyze_attention(payload: AnalyzeRequest):
    idx, tokens = prepare_input(payload.text)

    logits, diagnostics = model(idx, causal_mask=payload.causal_mask)

    layer = min(payload.layer, model.num_layers - 1)
    head = min(payload.head, model.num_heads - 1)
    mode = payload.analysis_mode.lower().strip()

    selected = diagnostics[layer]
    raw_attention = selected["attention_weights"][0, head]
    raw_scores = selected["raw_scores"][0, head]

    if mode == "raw":
        attention = raw_attention[: len(tokens), : len(tokens)]
        source_note = "Raw mode shows actual attention weights from the trained local PyTorch checkpoint. These weights are learned from a focused AttentionForge corpus, not from a large open-domain LLM."
        analysis_mode = "raw"
    else:
        attention = guided_attention_matrix(tokens, layer=layer, head=head, causal_mask=payload.causal_mask)
        source_note = "Guided mode creates an interpretable attention map for presentation while preserving masking and token-routing concepts. Raw mode uses the trained local PyTorch checkpoint."
        analysis_mode = "guided"

    probs = F.softmax(logits[0, -1], dim=-1)
    top_probs = torch.topk(probs, k=min(8, tokenizer.vocab_size))

    predictions = []
    for prob, idx_value in zip(top_probs.values, top_probs.indices):
        token = tokenizer.itos.get(int(idx_value), "<token>")
        predictions.append(
            {
                "token": token,
                "probability": round(float(prob), 5),
            }
        )

    return {
        "input_text": payload.text,
        "tokens": tokens,
        "sequence_length": len(tokens),
        "layer": layer,
        "head": head,
        "causal_mask": payload.causal_mask,
        "analysis_mode": analysis_mode,
        "source_note": source_note,
        "attention_matrix": matrix_to_list(attention),
        "raw_score_matrix": matrix_to_list(raw_scores[: len(tokens), : len(tokens)]),
        "top_connections": top_attention_links(attention, tokens),
        "next_token_predictions": predictions,
    }


@app.get("/api/claude-test")
def claude_test():
    """
    Lightweight endpoint to confirm whether the backend can see the Claude API key.
    This does not make a paid Claude API call.
    """
    debug = get_env_debug()
    return {
        "claude_configured": claude_is_configured(),
        "claude_model": get_claude_model(),
        "message": (
            "Claude API key is visible to the backend."
            if claude_is_configured()
            else "Claude API key is NOT visible to the backend. Check your .env file or environment variables."
        ),
        "debug": debug,
    }


def clean_raw_generation(text: str, max_sentences: int = 2) -> str:
    """Keep trained raw output readable in the UI."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return text

    # Keep only a few complete sentences when possible.
    parts = re.split(r"(?<=[.!?])\s+", text)
    complete = [part.strip() for part in parts if part.strip()]
    if len(complete) > max_sentences:
        text = " ".join(complete[:max_sentences])

    # Remove small accidental repeated word loops, ignoring punctuation.
    words = text.split()
    cleaned = []
    for word in words:
        normalized = re.sub(r"[^a-z0-9]", "", word.lower())
        prev = re.sub(r"[^a-z0-9]", "", cleaned[-1].lower()) if cleaned else None
        if prev and normalized and prev == normalized:
            continue
        cleaned.append(word)
    text = " ".join(cleaned)

    # Remove a few common short loop artifacts from tiny LM sampling.
    text = re.sub(r"\b(the user)\s+\1\b", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(the raw)\s+\1\b", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r",?\s*the claude the raw\.?$", ".", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+about\.$", " about the project.", text, flags=re.IGNORECASE)

    if text and not text.endswith((".", "?", "!")):
        text += "."
    return text


@app.post("/api/generate")
@torch.no_grad()
def generate_text(payload: GenerateRequest):
    mode = payload.mode.lower().strip()

    if mode == "claude":
        try:
            generated_text = generate_with_claude(
                seed_text=payload.seed_text,
                max_words=payload.max_new_tokens,
            )
            return {
                "mode": "claude",
                "seed_text": payload.seed_text,
                "generated_text": generated_text,
                "note": "Generated by Claude API. Raw PyTorch Transformer internals remain available in the workbench.",
            }
        except Exception as exc:
            fallback = coherent_generate(
                seed_text=payload.seed_text,
                max_new_tokens=payload.max_new_tokens,
            )
            return {
                "mode": "claude_fallback",
                "seed_text": payload.seed_text,
                "generated_text": fallback,
                "note": f"Claude API unavailable. The app used offline fallback instead. Error: {exc}",
            }

    if mode == "raw":
        ids = tokenizer.safe_encode(payload.seed_text)
        idx = torch.tensor([ids], dtype=torch.long)
        raw_token_budget = min(payload.max_new_tokens, 40)
        generated = model.generate_raw(
            idx,
            max_new_tokens=raw_token_budget,
            temperature=min(max(payload.temperature, 0.35), 0.65),
            top_k=1,
        )[0]

        raw_text = clean_raw_generation(tokenizer.decode(generated.tolist()))

        return {
            "mode": "raw",
            "seed_text": payload.seed_text,
            "generated_text": raw_text,
            "note": (
                "Generated by the trained local raw PyTorch Transformer checkpoint. "
                "It is trained on a focused AttentionForge corpus, so it gives project-specific answers rather than open-domain LLM answers."
                if CHECKPOINT_LOADED
                else "Checkpoint was not loaded, so raw mode is using random weights."
            ),
            "checkpoint_loaded": CHECKPOINT_LOADED,
        }

    generated_text = coherent_generate(
        seed_text=payload.seed_text,
        max_new_tokens=payload.max_new_tokens,
    )

    return {
        "mode": "coherent",
        "seed_text": payload.seed_text,
        "generated_text": generated_text,
        "note": "Coherent mode is designed for readable project demonstration while raw PyTorch mode remains available for internals.",
    }


@app.post("/api/explain-attention")
@torch.no_grad()
def explain_attention(payload: ClaudeExplainRequest):
    """
    Claude-powered natural-language explanation of the selected attention view.
    """
    analyze_payload = AnalyzeRequest(
        text=payload.input_text,
        layer=payload.layer,
        head=payload.head,
        causal_mask=payload.causal_mask,
        analysis_mode=payload.analysis_mode,
    )
    analyzed = analyze_attention(analyze_payload)

    try:
        explanation = explain_attention_with_claude(
            input_text=payload.input_text,
            tokens=analyzed["tokens"],
            top_connections=analyzed["top_connections"],
            layer=analyzed["layer"],
            head=analyzed["head"],
            causal_mask=analyzed["causal_mask"],
            analysis_mode=analyzed["analysis_mode"],
        )
        return {
            "mode": "claude",
            "explanation": explanation,
            "claude_configured": True,
        }
    except Exception as exc:
        fallback = (
            "Claude explanation is unavailable. The selected view shows token-to-token attention weights. "
            "Higher weights indicate stronger routing from the query token to the key token. "
            "Guided mode is designed for a cleaner interpretability demo, while raw mode shows actual untrained PyTorch weights."
        )
        return {
            "mode": "fallback",
            "explanation": fallback,
            "claude_configured": False,
            "error": str(exc),
        }


@app.get("/api/positional-encoding")
def positional_encoding(
    positions: int = Query(default=48, ge=4, le=96),
    dims: int = Query(default=48, ge=4, le=192),
):
    pe = model.positional_encoding.pe[0, :positions, :dims]
    return {
        "positions": positions,
        "dims": dims,
        "matrix": matrix_to_list(pe),
    }
