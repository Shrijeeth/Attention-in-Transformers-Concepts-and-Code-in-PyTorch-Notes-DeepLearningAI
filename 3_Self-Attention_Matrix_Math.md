# Self-Attention: Matrix Math Explained

## Learning Objectives

- Understand Q, K, V and why they are named that way.
- Derive the self-attention computation step-by-step from embeddings.
- Interpret dot products, scaling by sqrt(d_k), and row-wise softmax.
- Connect the math to the intuition of "which words should influence which".

## Q, K, V: What and Why

- **Definitions**
- Q = Query, K = Key, V = Value.
- Borrowed from database terminology:
  - The **query** is what you search with.
  - The **keys** are what you search over.
  - The **values** are what you retrieve when a key matches the query.

- **Database analogy (from the lesson)**
- Query: misspelled last name typed by Squatch (e.g., "Stammer").
- Keys: last names in the database (e.g., "Starmer").
- Value: the returned room number (e.g., 537). Bam!

## From Embeddings to Q, K, V

- **Start with token encodings**
- Each input token is turned into a numeric vector (embedding) and then augmented with positional encoding.
- For simple hand calculations, imagine each token is represented by 2 numbers. In practice, d_model is often 512 or larger.

- **Stack embeddings into a matrix**
- Let X be an n_tokens × d_model matrix of encoded tokens.

- **Linear projections to get Q, K, V**
- Use trainable weight matrices to project X into queries, keys, and values:

```text
Q = X · W_Q
K = X · W_K
V = X · W_V
```

- Shapes (typical):
- X: n × d_model
- W_Q: d_model × d_k  → Q: n × d_k
- W_K: d_model × d_k  → K: n × d_k
- W_V: d_model × d_v  → V: n × d_v

- **Note on PyTorch prints and transposes**
- Depending on how weights are stored/printed, you may see a transpose label on W to match the intended multiplication order. The key rule: ensure matrix dimensions align so the multiplication is valid. Small bam!

## Similarity via Dot Products (QK^T)

- **Score matrix**
- Compute pairwise similarities between each query and every key:

```text
S = Q · K^T    # S is n × n
```

- **Why K^T?**
- Dimensionally required so each query vector (row of Q) takes a dot product with every key vector (row of K) to yield an n-length vector of similarities.
- Semantically, each S[i, j] is the dot product similarity between token i's query and token j's key.

- **Dot product vs cosine similarity**
- Dot product is an unscaled similarity (can be any real number), closely related to cosine similarity (which is scaled to −1…1). The lesson uses dot products as unscaled similarities.

## Scaling by sqrt(d_k)

- **Stabilize magnitudes**
- Scale the score matrix by the square root of the key dimension d_k:

```text
S_scaled = S / sqrt(d_k)
```

- This simple scaling (by sqrt of values per token) was reported by the original authors to improve performance, even though it doesn’t impose a strict normalization. Small bam!

## Row-wise Softmax → Attention Weights

- **Convert similarities to percentages**
- Apply softmax to each row of S_scaled so each row sums to 1:

```text
A = softmax_rows(S_scaled)    # A is n × n
```

- **Interpretation**
- Row i of A contains how much each token j should influence token i’s new representation.
- Example intuition: for the sentence "the pizza … and it tasted good", the row for "it" should assign higher weight to "pizza" than to "oven". Bam!

## Weighted Sum of Values → Output Representations

- **Combine with V**
- Use attention weights to take a weighted sum of value vectors:

```text
O = A · V    # O is n × d_v
```

- **Meaning**
- Each output token representation O[i] is a mixture of value vectors, weighted by how relevant each token is to token i.
- This yields context-aware encodings for every token.

## Full Self-Attention Algorithm (Single Head)

1. Build token encodings X (embeddings + positional encodings).  
2. Project to Q, K, V with learned matrices W_Q, W_K, W_V.  
3. Compute unscaled similarities S = Q · K^T.  
4. Scale: S_scaled = S / sqrt(d_k).  
5. Row-wise softmax: A = softmax_rows(S_scaled).  
6. Output: O = A · V.  

Triple bam!

## Dimensions Cheat Sheet

- n: number of tokens.
- d_model: embedding size after positional encoding.
- d_k: size of query/key vectors (often = d_model / num_heads in multi-head attention).
- d_v: size of value vectors (often = d_k in simple settings).
- Shapes:
  - Q, K: n × d_k
  - V: n × d_v
  - S, A: n × n
  - O: n × d_v

## Intuition Recap

- Dot products measure how related tokens are (query ↔ key).
- Scaling tames magnitude for better training.
- Softmax turns similarities into attention percentages per token.
- Weighted sum of V produces the final context-aware representation for each token.

## Beginner's Guide to Scaled Dot-Product Self-Attention

A visual, simple walkthrough of the self-attention math with shapes, intuition, and references.

### Visual Overview

![Transformer architecture diagram (Wikimedia Commons)](https://commons.wikimedia.org/wiki/Special:FilePath/Transformer%2C_full_architecture.png)

- Source: [Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762)
- Visual walkthrough: [The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/)

### Symbols and Shapes (Single Head)

| Symbol | Meaning | Typical shape |
| --- | --- | --- |
| X | Encoded tokens (embeddings + positions) | n × d_model |
| W_Q, W_K, W_V | Learned projection weights | d_model × d_k (or d_v) |
| Q, K | Queries and Keys | n × d_k |
| V | Values | n × d_v |
| S = QK^T | Unscaled similarities | n × n |
| S_scaled = S/√d_k | Scaled similarities | n × n |
| A = softmax_rows(S_scaled) | Attention weights (per row sum = 1) | n × n |
| O = A·V | Output (context-aware) | n × d_v |

### The 5 Steps with Intuition

1. Project to Q, K, V: learn what to look for (Q), how to be found (K), and what to pass along (V).
2. Similarity S = QK^T: dot products say how related token i is to token j.
3. Scale by √d_k: keeps values numerically stable and training well-behaved.
4. Row-wise softmax: turn similarities into attention percentages per token.
5. Weighted sum O = A·V: mix values by those percentages to get context-aware outputs.

### Why Dot Product (vs Cosine)?

- Dot product is a simple, efficient similarity that correlates with cosine similarity but is unscaled.
- Scaling by √d_k compensates for larger magnitudes when d_k grows.

### Common Pitfalls (Beginner Notes)

- Forgetting K needs to be transposed in QK^T to align dimensions and compare each query with all keys.
- Misaligned shapes: always check n, d_model, d_k, d_v before multiplying.
- Softmax axis: apply along rows so each token’s weights sum to 1.

### Learn More / References

- Paper: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Visual guide: [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- Interactive: [Transformer Explainer (GPT‑2 attention)](https://poloclub.github.io/transformer-explainer/)
- Docs: [PyTorch MultiheadAttention](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)
- Course: [HF LLM Course — Architectures](https://huggingface.co/learn/llm-course/en/chapter1/6)
