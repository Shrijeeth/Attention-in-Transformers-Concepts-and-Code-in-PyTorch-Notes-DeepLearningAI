# Multi-Head Attention: Why and How

A beginner-friendly guide to multi-head attention (MHA): what it is, why it helps, how it’s wired into Transformers, and how dimensions are chosen in practice.

![Transformer architecture diagram (Wikimedia Commons)](https://commons.wikimedia.org/wiki/Special:FilePath/Transformer%2C_full_architecture.png)

- Paper: [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- Visual guide: [The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/)
- Tutorial: [UvA — Transformers and Multi-Head Attention](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html)
- Docs: [PyTorch nn.MultiheadAttention](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)

---

## 1) Motivation

- Single-head attention already captures token relations.
- Longer, complex sentences/paragraphs benefit from attending in multiple "ways" (subspaces) at the same time.
- Multi-head attention runs several attention "heads" in parallel, each with its own learned projections. Heads can specialize (e.g., syntax vs long-range dependency), leading to richer representations.

---

## 2) What is a "head"?

- A head is an independent scaled dot‑product attention unit with its own `W_Q^h, W_K^h, W_V^h`.
- Each head looks at the same input but through different projections, producing a different view of relationships.

### Per‑head math (single head `h`)

1. Project inputs to queries/keys/values for head `h`:

   - `Q_h = X · W_Q^h`
   - `K_h = X · W_K^h`
   - `V_h = X · W_V^h`

1. Scaled dot‑product attention (head `h`):

   - `S_h = Q_h · K_h^T`
   - `A_h = softmax_rows(S_h / sqrt(d_k))`
   - `O_h = A_h · V_h`

---

## 3) Putting heads together (concatenate + output projection)

- Compute `O_h` for each head `h = 1…H`.
- Concatenate head outputs on the feature dimension: `O_cat = concat(O_1, …, O_H)`.
- Project back to the model dimension with an output matrix `W_O`: `O = O_cat · W_O`.

This lets the model recombine diverse head features back into the original `d_model` size.

---

## 4) Shapes and common choices

Assume:

- `n`: number of tokens
- `d_model`: model width
- `H`: number of heads (e.g., 8 in the original paper)
- Typical choice: `d_k = d_v = d_model / H`

| Item | Shape |
| --- | --- |
| `X` (inputs per token) | `n × d_model` |
| `W_Q^h, W_K^h` | `d_model × d_k` |
| `W_V^h` | `d_model × d_v` |
| `Q_h, K_h` | `n × d_k` |
| `V_h` | `n × d_v` |
| `O_h = Attention(Q_h, K_h, V_h)` | `n × d_v` |
| `O_cat = concat_h O_h` | `n × (H·d_v)` |
| `W_O` | `(H·d_v) × d_model` |
| Final `O = O_cat · W_O` | `n × d_model` |

Notes:

- Using `d_k = d_v = d_model / H` keeps total compute roughly constant as we add heads.
- Heads can be computed in parallel for efficiency.

---

## 5) Connecting to the transcript examples

- Example: 3 heads, each producing 2 attention values → concatenation yields `6` values. A fully connected layer with 2 outputs brings it back to the original 2 encoded values.
- Alternative design: reduce per‑head output by using fewer `V` columns (e.g., `d_v = 1`), so each head outputs a single value. With original `d_model = 2`, two heads with `d_v = 1` already reconstruct `2` outputs without a large concatenation.
- In practice, Transformers use `W_O` to map the concatenated features back to `d_model` flexibly.

---

## 6) When and why MHA helps

- **Richer context**: different heads focus on different positions and relations.
- **Subspace specialization**: heads operate in distinct learned subspaces.
- **Stability**: scaling by `sqrt(d_k)` per head keeps magnitudes manageable.
- **Performance**: empirically improves translation and many NLP tasks; standard in modern LLMs.

---

## 7) Practical tips

- Start with `H` that divides `d_model` (common: 8 heads for `d_model=512`).
- Keep `d_k = d_v = d_model / H` unless you have a reason to vary.
- Use library implementations when possible: [PyTorch nn.MultiheadAttention](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html).
- For decoder blocks, use masked self‑attention for the first MHA; for encoder–decoder blocks, use cross‑attention MHA (queries from decoder, keys/values from encoder).

---

## 8) Summary

- Multi‑head attention = multiple parallel attention heads + concatenation + output projection.
- Enables learning diverse relationships simultaneously, improving representation power.
- Dimensioning (head counts and per‑head sizes) is chosen to balance capacity and compute.

---

## References

- Vaswani et al., 2017 — [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Jay Alammar — [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- UvA — [Transformers and Multi‑Head Attention](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html)
- D2L — [Multi-Head Attention](https://d2l.ai/chapter_attention-mechanisms-and-transformers/multihead-attention.html)
- PyTorch — [nn.MultiheadAttention](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)
