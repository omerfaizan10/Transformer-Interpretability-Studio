# AttentionForge Studio: A Professional Transformer Interpretability Lab

I built **AttentionForge Studio** to make Transformer internals easier to inspect.

Most people learn Transformers through diagrams or high-level APIs. Diagrams are useful, but they do not show the actual tensors. APIs are powerful, but they hide the mechanism.

AttentionForge Studio connects both sides.

It is a full-stack application with:

- a custom React frontend
- a FastAPI backend
- a raw PyTorch Transformer
- attention heatmaps
- causal masking controls
- positional encoding visualization
- generation modes

The project was created by **Omer Faizan**.

## Why This Project Exists

The goal is not to create a large language model.

The goal is to make the core Transformer mechanism visible.

A Transformer is built from understandable parts:

- token embeddings
- positional encoding
- queries
- keys
- values
- attention scores
- softmax weights
- feed-forward layers
- residual connections
- layer normalization

When these parts are hidden behind an API, the architecture can feel mysterious. This project exposes them directly.

## Why I Used React Instead of Streamlit

Streamlit is useful for quick ML demos, but a custom React interface gives more control over design and product quality.

With React, the project can feel like a real ML studio instead of a simple notebook demo.

The UI includes:

- a professional dark interface
- custom heatmaps
- polished cards
- clean typography
- token ribbons
- attention connection summaries
- architecture metrics

This makes the project stronger for GitHub, LinkedIn, and interviews.

## The Raw PyTorch Model

The backend uses a decoder-style Transformer written manually in PyTorch.

It avoids:

- Hugging Face
- `torch.nn.Transformer`
- `torch.nn.MultiheadAttention`

The attention mechanism is implemented directly.

The core idea is:

```python
scores = q @ k.transpose(-2, -1)
scores = scores / sqrt(head_dim)
attention_weights = softmax(scores)
context = attention_weights @ v
```

That is the heart of self-attention.

## Larger Transformer Preset

The upgraded model uses a more serious mini architecture:

- 6 Transformer layers
- 6 attention heads
- 192-dimensional token embeddings
- 768-dimensional feed-forward hidden layers
- roughly 2.9 million parameters

This is still small compared with modern language models, but it is much stronger as a learning and visualization model than a tiny beginner configuration.

## Attention Heatmaps

The main feature is the attention heatmap.

Rows represent query tokens.  
Columns represent key tokens.  
Cell values represent attention weights.

A bright cell means that one token is paying strong attention to another token.

This makes multi-head attention easier to understand because each head can be inspected separately.

## Causal Masking

For next-token prediction, the model must not look into the future.

Causal masking prevents future-token leakage.

The mask is lower triangular. A token can attend to itself and previous tokens, but not future tokens.

This is one of the most important ideas behind decoder-only language models.

## Meaningful Generation

The earlier version generated gibberish because the model started with random weights.

That is expected behavior for an untrained Transformer.

In this version, I added a coherent generation mode so the UI produces readable, relevant sentences by default. The raw untrained Transformer mode is still available for debugging and demonstration.

This makes the project better for presentation because the generation tab no longer creates a bad first impression.

## Why This Project Is Portfolio-Worthy

AttentionForge Studio shows more than just model code.

It demonstrates:

- machine learning fundamentals
- raw PyTorch implementation ability
- backend API design
- frontend engineering
- explainability thinking
- product presentation
- technical communication

That combination makes it stronger than a basic beginner notebook.

## Final Thought

Attention is easier to understand when you can see it.

AttentionForge Studio turns the Transformer into something explorable instead of abstract.

## Update: Trained Raw Transformer Mode

The latest version includes a trained local raw PyTorch checkpoint. Earlier prototypes used random weights in raw mode, which made generation look unstable. That was useful for proving the architecture existed, but it was not strong enough for a polished demo.

The new raw mode loads a checkpoint trained on a focused AttentionForge corpus. It can now generate coherent project-specific text about attention, masking, positional encoding, token routing, and model explanation. Claude remains available as a narration layer, but the local Transformer is no longer random.

This keeps the project honest: the raw model is not a general LLM, but it is a real trained Transformer language model for the studio domain.
