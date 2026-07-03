"""
Claude API integration for AttentionForge Studio.

Important:
- The Claude API key must stay on the backend.
- Never put ANTHROPIC_API_KEY in the React frontend.
- Use a local .env file or hosting environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Robust Windows-friendly .env loading:
# backend/attentionforge/claude_client.py -> project root is parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"

# Load the project-root .env first, then also allow normal environment variables.
load_dotenv(dotenv_path=ENV_PATH, override=True)
load_dotenv(override=True)


def get_env_debug() -> dict[str, Any]:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    return {
        "project_root": str(PROJECT_ROOT),
        "env_path": str(ENV_PATH),
        "env_file_exists": ENV_PATH.exists(),
        "anthropic_key_visible": bool(key),
        "anthropic_key_preview": f"{key[:8]}..." if key else "",
        "claude_model": get_claude_model(),
    }


def claude_is_configured() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def get_claude_model() -> str:
    return os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5")


def _extract_text(message: Any) -> str:
    parts = []
    for block in getattr(message, "content", []):
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts).strip()


def generate_with_claude(seed_text: str, max_words: int = 140) -> str:
    if not claude_is_configured():
        raise RuntimeError(
            f"Claude API key not configured. Expected .env at {ENV_PATH} or ANTHROPIC_API_KEY in environment."
        )

    from anthropic import Anthropic

    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    prompt = f"""
You are writing output for a portfolio project called AttentionForge Studio.

User seed text:
{seed_text!r}

Write a polished, useful, non-repetitive response around the seed text.
Requirements:
- 3 to 5 sentences maximum.
- Make the response meaningful for the actual seed topic.
- Also connect it lightly to how AttentionForge Studio tokenizes text and visualizes Transformer attention.
- Do not claim the raw PyTorch mini Transformer is a full chatbot or pretrained LLM.
- Do not repeat the same sentence.
- Do not use markdown headings.
- Keep it under {max_words} words.
"""

    message = client.messages.create(
        model=get_claude_model(),
        max_tokens=420,
        temperature=0.7,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    text = _extract_text(message)
    if not text:
        raise RuntimeError("Claude returned an empty response.")

    return text


def explain_attention_with_claude(
    input_text: str,
    tokens: list[str],
    top_connections: list[dict[str, Any]],
    layer: int,
    head: int,
    causal_mask: bool,
    analysis_mode: str,
) -> str:
    if not claude_is_configured():
        raise RuntimeError(
            f"Claude API key not configured. Expected .env at {ENV_PATH} or ANTHROPIC_API_KEY in environment."
        )

    from anthropic import Anthropic

    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    compact_links = []
    for item in top_connections[:8]:
        compact_links.append(
            f"{item.get('query_index')}:{item.get('query_token')} -> "
            f"{item.get('key_index')}:{item.get('key_token')} "
            f"({round(float(item.get('weight', 0)) * 100, 1)}%)"
        )

    prompt = f"""
You are explaining a Transformer attention visualization in AttentionForge Studio.

Input text:
{input_text}

Tokens:
{tokens}

Selected layer: {layer}
Selected head: {head}
Causal mask enabled: {causal_mask}
Analysis mode: {analysis_mode}

Top attention connections:
{compact_links}

Write a concise explanation for a portfolio demo.
Requirements:
- 4 bullet points maximum.
- Explain what the selected head appears to focus on.
- Explain self-attention and repeated/anchor-token behavior if visible.
- Mention that guided mode is for interpretability if analysis_mode is guided.
- Do not overclaim that the mini Transformer understands language like a pretrained LLM.
"""

    message = client.messages.create(
        model=get_claude_model(),
        max_tokens=500,
        temperature=0.4,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    text = _extract_text(message)
    if not text:
        raise RuntimeError("Claude returned an empty explanation.")

    return text
