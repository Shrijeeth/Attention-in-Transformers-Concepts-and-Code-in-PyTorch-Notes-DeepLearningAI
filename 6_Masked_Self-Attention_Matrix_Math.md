# Masked Self-Attention: Matrix Math Explained

A beginner-friendly, step-by-step walkthrough of the matrix math behind masked (causal) self-attention, with shapes, intuition, examples, and references.

![Transformer architecture diagram (Wikimedia Commons)](https://commons.wikimedia.org/wiki/Special:FilePath/Transformer%2C_full_architecture.png)

- Paper: [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- Visual guide: [The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/)
- Interactive: [Transformer Explainer (GPT‑2 attention)](https://poloclub.github.io/transformer-explainer/)
- Tutorial: [Understanding and Coding Self-Attention (Raschka)](https://magazine.sebastianraschka.com/p/understanding-and-coding-self-attention)

---

## What changes vs normal self-attention?

- Normal self-attention: each token can attend to tokens before and after it.
- Masked (causal) self-attention: each token can only attend to itself and tokens up to its position (no look-ahead).
- Mathematically, the difference is simple: we add a **mask matrix** `M` to the scaled similarities before softmax.

---

## Symbols and shapes (single head)

| Symbol | Meaning | Typical shape |
| --- | --- | --- |
| `X` | Encoded tokens (embeddings + positions) | `n × d_model` |
| `W_Q, W_K, W_V` | Learned projection weights | `d_model × d_k` (or `d_v`) |
| `Q = X·W_Q` | Queries | `n × d_k` |
| `K = X·W_K` | Keys | `n × d_k` |
| `V = X·W_V` | Values | `n × d_v` |
| `S = Q·K^T` | Unscaled similarities | `n × n` |
| `S_scaled = S / sqrt(d_k)` | Scaled similarities | `n × n` |
| `M` | Mask (0 for allowed, −∞ for disallowed) | `n × n` |
| `S_masked = S_scaled + M` | Masked similarities | `n × n` |
| `A = softmax_rows(S_masked)` | Attention weights | `n × n` |
| `O = A·V` | Output (context-aware) | `n × d_v` |

---

## Step-by-step derivation (with intuition)

Given token encodings `X` (embeddings + positional encoding):

1. Project to queries, keys, values: `Q = X·W_Q`, `K = X·W_K`, `V = X·W_V`.
1. Compare each query with all keys: `S = Q·K^T`.
1. Scale to stabilize magnitudes: `S_scaled = S / sqrt(d_k)`.
1. Add the mask: `S_masked = S_scaled + M`.
1. Row-wise softmax to get percentages: `A = softmax_rows(S_masked)`.
1. Weighted sum of values: `O = A·V`.

Why it works:

- The mask `M` has zeros where attention is allowed and large negative numbers (theoretically −∞) where attention is disallowed.
- Adding −∞ ensures those positions become zero after softmax.

---

## Example: prompt “write a poem” (3 tokens)

Tokens: `[write, a, poem]` → after embedding and positional encoding → `X`.

Causal mask `M` (allow only current and previous positions):

```text
M = [[  0, -∞, -∞],
     [  0,   0, -∞],
     [  0,   0,   0]]
```

Effect on attention rows after softmax:

- Row for `write`: can only attend to `write` → 100% on itself.
- Row for `a`: can attend to `write` and `a` → nonzero on first two entries, 0 on `poem`.
- Row for `poem`: can attend to all three tokens.

Practical note: in code, we typically use a large negative constant (e.g., `-1e9`) instead of literal `-∞` for numerical stability.

---

## Minimal PyTorch-style pseudocode (single head)

```python
# X: [n, d_model]
Q = X @ W_Q  # [n, d_k]
K = X @ W_K  # [n, d_k]
V = X @ W_V  # [n, d_v]

S = Q @ K.T                      # [n, n]
S_scaled = S / (d_k ** 0.5)      # [n, n]
S_masked = S_scaled + M          # [n, n], M has 0 or large negative values
A = softmax(S_masked, dim=1)     # [n, n] rows sum to 1
O = A @ V                        # [n, d_v]
```

To build `M` for causal masking, use a lower-triangular mask (including the diagonal). In batched/multi-head settings, broadcast `M` to the appropriate dimensions.

---

## Why masked attention enables generation

- Training objective: predict the next token given the prefix (teacher forcing).
- Causal mask prevents peeking at future tokens.
- At inference: generate token by token, feeding outputs back as inputs.

---

## Troubleshooting and common pitfalls

- Ensure `K` is transposed in `Q·K^T` to compare each query with every key.
- Use the correct softmax axis (rows), so each row’s weights sum to 1.
- Replace `−∞` with a large negative constant (e.g., `-1e9`) to avoid NaNs.
- Verify all shapes before matmul: `Q,K: [n,d_k]`, `V: [n,d_v]`, `S,A: [n,n]`, `O: [n,d_v]`.

---

## References and further reading

- Vaswani et al., 2017 — [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Jay Alammar — [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- Hugging Face — [Transformer Architectures](https://huggingface.co/learn/llm-course/en/chapter1/6)
- Raschka — [Understanding and Coding Self-Attention](https://magazine.sebastianraschka.com/p/understanding-and-coding-self-attention)
- GeeksforGeeks — [How Do Self-Attention Masks Work?](https://www.geeksforgeeks.org/nlp/how-do-self-attention-masks-work/)
