# Encoder–Decoder Attention (Cross-Attention): What, Why, and Where the Names Come From

A beginner-friendly guide to encoder–decoder attention (also called cross-attention): how it works, why it matters, and how it connects the encoder and decoder in the original Transformer.

![Transformer architecture diagram (Wikimedia Commons)](https://commons.wikimedia.org/wiki/Special:FilePath/Transformer%2C_full_architecture.png)

- Paper: [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- Visual guide: [The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/)
- Course: [Hugging Face — Transformer Architectures](https://huggingface.co/learn/llm-course/en/chapter1/6)
- Tutorial: [UvA — Transformers and Multi-Head Attention](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html)

---

## 1) Quick recap: self-attention and masked self-attention

- **Encoder-only (self-attention)**
  - Learns context-aware embeddings by attending bidirectionally (left and right) over the input.
  - Great for understanding tasks: classification, retrieval, semantic similarity, embeddings.

- **Decoder-only (masked/causal self-attention)**
  - Attends only to current and past tokens (no look-ahead) to generate the next token.
  - Great for generation: chat, code, long-form text.

### Three attention flavors at a glance

| Attention | Where it lives | What attends to what | Masking | Typical use |
| --- | --- | --- | --- | --- |
| Self-attention | Encoder | Input tokens ↔ input tokens (bidirectional) | No look-ahead mask | Understanding/encoding |
| Masked self-attention | Decoder | Output tokens ↔ previous output tokens | Causal (lower-triangular) | Generation |
| Encoder–decoder (cross) attention | Decoder | Decoder queries ↔ encoder outputs | Usually no look-ahead mask on cross links | Mapping input → output |

---

## 2) Where the names come from

- The original Transformer has two parts:
  - An **encoder** that uses self-attention to build a representation of the input.
  - A **decoder** that uses masked self-attention to generate outputs and cross-attends to the encoder outputs.
- Later, practitioners realized:
  - You can use just the encoder → **encoder-only** models (e.g., BERT) for understanding.
  - You can use just the decoder → **decoder-only** models (e.g., GPT) for generation.

---

## 3) How encoder–decoder attention works (step by step)

Intuition: The decoder wants to ask the encoder, “What parts of the input are relevant to my current generation step?” Cross-attention computes this by matching decoder queries against encoder keys and then mixing encoder values.

- Setup
  - Encoder output (contextualized input): `Z_enc ∈ R^{n_src × d_model}`
  - Decoder hidden states (after masked self-attn): `Z_dec ∈ R^{n_tgt × d_model}`

### Cross-attention math (single head)

1. Compute projections

   - Queries from decoder: `Q = Z_dec · W_Q`  
   - Keys from encoder: `K = Z_enc · W_K`  
   - Values from encoder: `V = Z_enc · W_V`

1. Similarities and scaling

   - `S = Q · K^T`  
   - `S_scaled = S / sqrt(d_k)`

1. Row-wise softmax → attention weights

   - `A = softmax_rows(S_scaled)`

1. Weighted sum of encoder values

   - `O = A · V`

### Shapes and symbols

| Symbol | Meaning | Shape |
| --- | --- | --- |
| `Z_enc` | Encoder outputs | `n_src × d_model` |
| `Z_dec` | Decoder states (post masked self-attn) | `n_tgt × d_model` |
| `W_Q, W_K, W_V` | Projection weights | `d_model × d_k` / `d_model × d_v` |
| `Q` | Decoder queries | `n_tgt × d_k` |
| `K` | Encoder keys | `n_src × d_k` |
| `V` | Encoder values | `n_src × d_v` |
| `S = QK^T` | Similarities | `n_tgt × n_src` |
| `A = softmax_rows(S/√d_k)` | Attention weights | `n_tgt × n_src` |
| `O = AV` | Cross-attended outputs | `n_tgt × d_v` |

Notes:

- Cross-attention typically does not use a causal mask because it connects each decoder position to the entire encoded input. The decoder’s own masked self-attention already prevents look-ahead across generated tokens.
- In multi-head attention, the same procedure runs in parallel across heads, then concatenations and linear projections follow.

---

## 4) Example: translation (“Pizza is great” → “¡La pizza es genial!”)

- Encoder (self-attention) ingests the source sentence and produces contextual embeddings capturing meaning and word relations.
- Decoder step t = 1 (generating the first target token):
  - Decoder masked self-attn builds a state using previous outputs (none yet at t=1) and positions.
  - Cross-attention matches the decoder query against encoder outputs to focus on relevant source tokens.
- As t increases, the decoder repeats masked self-attn + cross-attn + feedforward to generate the next target token until end-of-sequence.

---

## 5) Practical tips and FAQs

- Do I need cross-attention for classification?
  - No. Encoder-only models often suffice (use [CLS] pooling or mean pooling).
- Do I need cross-attention for generation?
  - Not always. Decoder-only models generate well without an encoder; cross-attention is crucial when conditioning generation tightly on an input (translation, summarization, instruction-following in encoder–decoder setups).
- Why both self-attn and cross-attn in a decoder?
  - Masked self-attn ensures causal generation; cross-attn lets the decoder consult the encoded source at every step.

---

## 6) Model types and common tasks

| Family | Attention inside | Typical tasks | Example models |
| --- | --- | --- | --- |
| Encoder-only | Self-attention | Classification, retrieval, embeddings, NER | BERT, RoBERTa |
| Decoder-only | Masked self-attention | Next-token generation, chat, code completion | GPT, Llama, Mistral |
| Encoder–Decoder | Encoder self-attn + Decoder masked self-attn + Cross-attn | Translation, summarization, seq2seq tasks | T5, original Transformer |

---

## 7) Beyond language: multimodal cross-attention

Cross-attention also powers multimodal systems:

- Vision encoder → text decoder (image captioning, VQA)
- Audio encoder → text decoder (ASR prompts, audio QA)
- General pattern: an encoder produces modality-specific embeddings, and a text decoder cross-attends to generate language outputs.

Reading pointers:

- Survey the pattern in the HF course: [Transformer Architectures](https://huggingface.co/learn/llm-course/en/chapter1/6)
- UvA tutorial’s attention sections: [Transformers and Multi-Head Attention](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html)

---

## 8) Summary

- The original Transformer introduced a two-part architecture (encoder + decoder) connected via **encoder–decoder (cross) attention**.
- Cross-attention uses decoder-derived queries and encoder-derived keys/values to align target generation with source meaning.
- Modern practice distilled variants: **encoder-only** for understanding and **decoder-only** for generation; **encoder–decoder** remains strong for seq2seq and multimodal tasks.

---

## References

- Vaswani et al., 2017 — [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Jay Alammar — [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- Hugging Face — [Transformer Architectures](https://huggingface.co/learn/llm-course/en/chapter1/6)
- UvA — [Transformers and Multi-Head Attention](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html)
