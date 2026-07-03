"""
Hashing word tokenizer for AttentionForge Studio.

Why hashing?
- A normal word-level tokenizer shows <unk> for words outside the tiny demo corpus.
- That looks bad in the UI when users type their own text.
- This tokenizer keeps the original words for display and maps unseen words to stable hash buckets internally.

The model still receives valid token IDs, while the UI shows the user's real words.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re


DEFAULT_CORPUS = """
attention forge studio is a transformer interpretability workbench built with raw pytorch, fastapi, react, and a claude narration layer.
the raw model is trained on a focused corpus about attention, masking, positional encoding, token routing, and model explanations.
this project is not a general chatbot; it is a transparent laboratory for visualizing transformer internals.

attention is a routing mechanism that lets each token decide which earlier tokens matter.
attention weights are calculated from queries and keys, then used to mix value vectors.
queries describe what a token is looking for, keys describe what each token offers, and values carry the information.
multi head attention gives the model several relationship views at the same time.
one attention head may focus on local context while another head may focus on the first anchor token or repeated words.

a transformer turns text into vectors, adds positional encoding, applies multi head attention, and then uses a feed forward network.
residual connections keep information flowing through the network, while layer normalization stabilizes the representation.
stacking transformer blocks lets token representations become more contextual layer by layer.

a causal mask prevents a decoder model from looking at future tokens during next token prediction.
with the mask enabled, each row can only attend to itself and earlier columns.
without a causal mask, future tokens leak information and the language modeling task becomes dishonest.

a positional encoding gives the model a sense of order because attention alone is order agnostic.
the positional heatmap shows sine and cosine patterns across positions and embedding dimensions.
the attention heatmap shows token to token routing across the selected layer and head.

bottle is an example input. the raw transformer treats bottle as a token and predicts a continuation from the training distribution.
glass is an example input. the studio can show how glass attends to nearby tokens and earlier context.
earth is an example input. the system can explain that earth is being tokenized, embedded, and routed through attention.
home is an example input. the token home can attend to earlier words when causal masking is enabled.
help is an example input. the studio can answer by explaining the workbench, the heatmap, and the raw model.

when the user enters earth is my home, the first token can only attend to itself, and later tokens can attend to earlier context.
when the user enters bottle on table, the model visualizes how the selected head routes token information.
when the user enters glass of water, the model can show attention links between the object and its context.
when the user enters explain attention, the model can produce a concise explanation about token routing.
when the user enters causal mask, the model can explain why future tokens are blocked.
when the user enters positional encoding, the model can explain how order is injected into token vectors.

raw mode uses the trained pytorch checkpoint for local generation.
claude mode uses the claude api for polished natural language explanations.
guided mode creates a cleaner interpretability map for presentation, while raw mode exposes actual model weights.
for a portfolio demo, the strongest claim is that the project combines a custom transformer engine with an explanation layer.

the backend exposes raw pytorch tensors such as attention matrices, raw scores, next token probabilities, and top token routes.
the frontend turns tensor outputs into readable visual explanations with heatmaps, cards, and token ribbons.
the user can select a layer and head to inspect how information moves through the sequence.
the model status panel reports whether the trained checkpoint is loaded and whether claude is configured.

a good explanation should be honest: the mini transformer is trained for a focused studio demo, not for open domain question answering.
raw generation is useful for showing that the model learned the project corpus, while claude narration is useful for flexible language.
attention forge studio is designed to look professional, run locally, and demonstrate understanding of transformer mechanics.
"""

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>"]
HASH_BUCKETS = 96


def basic_tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z]+|[0-9]+|[^\w\s]", text.lower())


def stable_bucket(token: str, bucket_count: int = HASH_BUCKETS) -> int:
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return int(digest, 16) % bucket_count


@dataclass
class WordTokenizer:
    vocab: list[str]
    stoi: dict[str, int]
    itos: dict[int, str]
    hash_tokens: list[str]

    @classmethod
    def from_text(cls, text: str) -> "WordTokenizer":
        corpus_tokens = basic_tokenize(text)
        known_tokens = sorted(set(corpus_tokens))
        hash_tokens = [f"<hash_{i:02d}>" for i in range(HASH_BUCKETS)]
        vocab = SPECIAL_TOKENS + known_tokens + hash_tokens

        stoi = {token: i for i, token in enumerate(vocab)}
        itos = {i: token for token, i in stoi.items()}

        return cls(vocab=vocab, stoi=stoi, itos=itos, hash_tokens=hash_tokens)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def display_tokens(self, text: str) -> list[str]:
        """
        Returns actual user-facing tokens.
        This prevents the UI from showing <unk>.
        """
        return basic_tokenize(text)

    def encode(self, text: str, add_special: bool = False) -> list[int]:
        tokens = basic_tokenize(text)
        ids = []

        for token in tokens:
            if token in self.stoi:
                ids.append(self.stoi[token])
            else:
                bucket_token = self.hash_tokens[stable_bucket(token)]
                ids.append(self.stoi[bucket_token])

        if add_special:
            ids = [self.stoi["<bos>"]] + ids + [self.stoi["<eos>"]]

        return ids

    def safe_encode(self, text: str, fallback: str = "attention") -> list[int]:
        ids = self.encode(text)
        if not ids:
            ids = self.encode(fallback)
        return ids

    def decode(self, ids) -> str:
        tokens = [self.itos.get(int(i), "<hash>") for i in ids]
        return detokenize(tokens)


def detokenize(tokens: list[str]) -> str:
    clean = []

    for token in tokens:
        if token in SPECIAL_TOKENS or token.startswith("<hash_"):
            continue

        if token in [".", ",", ":", ";", "?", "!"]:
            if clean:
                clean[-1] = clean[-1] + token
        else:
            clean.append(token)

    text = " ".join(clean)
    if text:
        text = text[0].upper() + text[1:]
    return text
