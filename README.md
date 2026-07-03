# AttentionForge Studio

**AttentionForge Studio** is a professional full-stack Transformer interpretability lab built with a custom React interface, FastAPI backend, and a raw PyTorch Transformer implemented from scratch.

Created by **Omer Faizan**.

No Hugging Face.  
No `torch.nn.Transformer`.  
No `torch.nn.MultiheadAttention`.

This is not positioned as a toy chatbot. It is a visual research-style studio for understanding how Transformer internals work.

---

## V2 Fixes

This version fixes three important demo issues:

- Custom input no longer displays `<unk>` because unseen words use stable hash buckets internally while the UI shows the real words.
- Generation now changes based on the seed text in coherent mode.
- Attention analysis has two modes: **Guided interpretability mode** for clean portfolio demos and **Raw PyTorch mode** for actual untrained model weights.

## What Makes This Version Better

This upgraded version improves the earlier project in four ways:

1. **More professional UI**
   - Custom React + Vite frontend
   - Clean dark studio design
   - Better layout, typography, spacing, and visual hierarchy
   - Less “AI-generated dashboard” feel

2. **Better project name**
   - Renamed from AttentionLens to **AttentionForge Studio**
   - Stronger, more original, more portfolio-friendly branding

3. **Larger Transformer preset**
   - 6 decoder-style Transformer layers
   - 6 attention heads
   - 192-dimensional embeddings
   - 768 hidden units in feed-forward layers
   - About 2.9M parameters, depending on vocabulary size

4. **Meaningful generation**
   - Default generation uses a coherent domain-guided mode so the output reads like real sentences
   - Raw Transformer generation is still available as a research/debug mode
   - This avoids the bad first impression of random untrained gibberish while keeping the raw PyTorch internals visible

---

## Core Features

- Multi-head attention heatmaps
- Layer and head selector
- Causal masking switch
- Word-level token ribbon
- Top attention connection table
- Positional encoding explorer
- Next-token prediction panel
- Coherent generation mode
- Raw Transformer generation mode
- Full backend API
- Raw PyTorch implementation

---

## Architecture

```text
attentionforge-transformer-studio/
├── README.md
├── blog_post.md
├── requirements.txt
├── .gitignore
├── backend/
│   ├── main.py
│   └── attentionforge/
│       ├── __init__.py
│       ├── generator.py
│       ├── model.py
│       └── tokenizer.py
├── frontend/
│   ├── package.json
│   ├── vite.config.js
│   ├── index.html
│   └── src/
│       ├── App.jsx
│       ├── api.js
│       ├── index.css
│       ├── main.jsx
│       └── components/
│           ├── Heatmap.jsx
│           ├── MetricCard.jsx
│           └── TokenRibbon.jsx
├── notebooks/
│   └── raw_pytorch_transformer_core.ipynb
└── assets/
    └── project_card.md
```

---


## V3 Generation Fix

The generation tab now uses a cleaner seed-aware explanation mode.

Example:

```text
glass — The seed "glass" is treated as the focus input for this interpretability demo. AttentionForge Studio does not claim to be a general chatbot; it shows how input tokens are routed through Transformer-style attention.
```

This avoids repeated template text and keeps the project honest: the live demo is an interpretability studio, not a fake chatbot.


## V4 Claude API Upgrade

This version adds optional Claude API support.

The project now has three generation modes:

- `Claude API mode`: high-quality natural-language output using your Claude API key
- `Offline coherent fallback`: no API key required, but limited
- `Raw Transformer mode`: samples directly from the untrained raw PyTorch mini Transformer

Claude is also used for the optional **Claude Narrator** in the attention workbench. It explains the selected layer, head, mask setting, and top token routes in plain English.

### Local Claude Setup

Create a `.env` file in the project root:

```text
ANTHROPIC_API_KEY=your_real_claude_api_key_here
CLAUDE_MODEL=claude-haiku-4-5
```

Then run:

```bash
python -m uvicorn backend.main:app --reload --port 8000
```

Important: never put your Claude API key inside the React frontend and never commit `.env` to GitHub.

### Deployment Setup

On Render/Railway, add these environment variables:

```text
ANTHROPIC_API_KEY=your_real_claude_api_key_here
CLAUDE_MODEL=claude-haiku-4-5
```

Use `claude-sonnet-4-6` if you want stronger output and your API key has access.


## Claude Troubleshooting

If Claude API mode still gives offline fallback text, the backend is not seeing your API key.

Check this URL while the backend is running:

```text
http://127.0.0.1:8000/api/claude-test
```

You should see:

```json
{
  "claude_configured": true
}
```

If it says `false`, create a file named exactly `.env` in the main project folder:

```text
ANTHROPIC_API_KEY=your_real_api_key_here
CLAUDE_MODEL=claude-haiku-4-5
```

Then stop and restart the backend.

Common Windows mistake: the file is accidentally named `.env.txt`. In File Explorer, enable file extensions to confirm the filename.


## V6 Environment Check

Run this before starting the backend:

```bash
python check_claude_env.py
```

It prints the exact `.env` path the backend expects and whether `ANTHROPIC_API_KEY` is visible.

The `.env` file must be in the main project folder, the same folder as:

```text
README.md
requirements.txt
backend
frontend
check_claude_env.py
```


## V7 Trained Raw Transformer Upgrade

This version adds a real trained local raw PyTorch checkpoint for Raw Transformer mode.

What changed:

- Raw Transformer mode now loads `backend/checkpoints/raw_transformer_trained.pt`.
- The checkpoint is trained on a focused AttentionForge corpus about attention, masking, positional encoding, token routing, and model explanations.
- Raw mode now produces coherent project-specific output instead of random text.
- Claude mode remains available for polished natural-language explanations.
- The attention matrix UI was compacted so small inputs no longer create oversized boxes.
- The model output tab and button labels were cleaned up to make the app feel more professional.

Important positioning:

```text
Raw Transformer mode = trained local PyTorch checkpoint for focused project-specific output
Claude API mode = high-quality explanation layer for arbitrary language
Workbench = attention internals, heatmaps, masking, layer/head inspection
```

The raw checkpoint is not a general-purpose LLM. It is intentionally trained on a compact project corpus so it can demonstrate learned next-token prediction and attention behavior locally.

To retrain the checkpoint:

```bash
python backend/train_raw_checkpoint.py
```

## Quick Start

### 1. Start backend

From the project root:

```bash
pip install -r requirements.txt
python -m uvicorn backend.main:app --reload --port 8000
```

### 2. Start frontend

Open a second terminal:

```bash
cd frontend
npm install
npm run dev
```

### 3. View UI

Open:

```text
http://localhost:5173
```

---

## API Endpoints

### Health

```text
GET /api/health
```

### Config

```text
GET /api/config
```

### Analyze Attention

```text
POST /api/analyze
```

Example:

```json
{
  "text": "attention lets each token decide which earlier tokens matter",
  "layer": 0,
  "head": 0,
  "causal_mask": true
}
```

### Generate

```text
POST /api/generate
```

Example:

```json
{
  "seed_text": "attention",
  "max_new_tokens": 120,
  "temperature": 0.8,
  "mode": "coherent"
}
```

Modes:

- `coherent`: readable studio demo mode
- `raw`: pure untrained Transformer sampling mode

---

## Suggested GitHub Description

> Professional full-stack Transformer interpretability studio using React, FastAPI, and raw PyTorch. Visualizes multi-head attention, causal masking, positional encoding, and generation behavior without Hugging Face.

---

## Resume Bullet

Built **AttentionForge Studio**, a full-stack Transformer interpretability platform using React, FastAPI, and raw PyTorch. Implemented multi-head self-attention, causal masking, sinusoidal positional encoding, and autoregressive generation from scratch, then exposed model internals through a professional attention visualization interface.

---

## Important Positioning

This project should be presented as:

> A Transformer interpretability and education studio.

Not as:

> A chatbot.

The generation tab is included to demonstrate autoregressive behavior, but the strongest part of the project is the visual explanation of attention internals.

---

## Push to GitHub

```bash
git init
git add .
git commit -m "Add AttentionForge Studio"

git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/attentionforge-transformer-studio.git
git push -u origin main
```
