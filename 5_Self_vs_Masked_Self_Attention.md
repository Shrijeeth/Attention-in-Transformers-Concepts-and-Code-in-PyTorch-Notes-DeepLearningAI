# Self-Attention vs Masked Self-Attention: Strengths, Weaknesses, and Use Cases

A deep dive into when and why to use self-attention or masked (causal) self-attention, how embeddings and positional encoding fit in, and what this means for encoder-only vs decoder-only Transformers.

![Transformer architecture diagram (Wikimedia Commons)](https://commons.wikimedia.org/wiki/Special:FilePath/Transformer%2C_full_architecture.png)

- Paper: [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- Visual guide: [The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/)
- Interactive: [Transformer Explainer (GPT‑2 attention)](https://poloclub.github.io/transformer-explainer/)

---

## 1) Quick Recap: What Attention Does

- **Goal**
  - Establish relationships among tokens in a sequence.

- **Example**
  - "The pizza came out of the oven and it tasted good."
  - Attention should associate "it" → "pizza" (not "oven").

---

## 2) From Words to Numbers: Embeddings

Static word embeddings predate Transformers and map words to vectors such that similar words ("great", "awesome") have similar vectors.

- **Why not random numbers?**
  - Random assignment makes related words unrelated numerically, requiring more data/complexity to learn semantics.

- **Static embedding methods**
  - Word2Vec (CBOW/Skip-gram): [Mikolov et al., 2013](https://arxiv.org/abs/1301.3781)
  - GloVe: [Pennington, Socher, Manning, 2014](https://nlp.stanford.edu/pubs/glove.pdf)

- **Toy embedding network (as in the lesson)**
  - Inputs: one feature per unique word.
  - Hidden: 2 activation units → yields 2 embedding values per word.
  - Objective: predict next word(s) so that words used in similar contexts learn similar embeddings.

### Static vs Contextual Embeddings

| Aspect | Static word embeddings (Word2Vec/GloVe) | Contextual embeddings (Transformer encoders) |
| --- | --- | --- |
| Depends on context? | No | Yes (self-attention looks at full sequence) |
| Representation | One vector per word type | One vector per token instance |
| Captures polysemy | Limited | Strong (different senses in different contexts) |
| Typical use | Features for downstream models | Universal encoders for many tasks |

---

## 3) Order Matters: Positional Encoding

Self-attention alone is order-agnostic. Transformers inject order using positional encodings (often sinusoids) added to embeddings.

- Explainer: [Transformer Positional Encoding (Kazemnejad)](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/)

---

## 4) Self-Attention vs Masked Self-Attention

### Self-Attention (bidirectional within a sequence)

- Each token can attend to tokens before and after it.
- Used in encoders to compute **contextual embeddings**.
- Great for understanding tasks where full context is available (classification, retrieval, semantic similarity).

### Masked (Causal) Self-Attention

- Each token can only attend to tokens up to its own position (no looking ahead).
- Enforced via a triangular "look-ahead" mask.
- Used in decoders for **autoregressive generation** (next-token prediction).
- Guides the model to generate text step-by-step without "cheating" on future tokens.

Further reading on masks: [How Do Self-Attention Masks Work?](https://www.geeksforgeeks.org/nlp/how-do-self-attention-masks-work/) · [Raschka: Understanding and Coding Self-Attention](https://magazine.sebastianraschka.com/p/understanding-and-coding-self-attention)

---

## 5) Encoder-Only vs Decoder-Only (and Why It Matters)

| Model family | Attention type | Looks at | Trains to | Best for | Examples |
| --- | --- | --- | --- | --- | --- |
| Encoder-only | Self-attention | Full context (bidirectional) | Understand inputs | Embeddings, classification, retrieval, clustering | BERT, RoBERTa |
| Decoder-only | Masked self-attention | Past context only (causal) | Generate next token | Text generation, chat, code completion | GPT, Llama, Mistral |
| Encoder–Decoder | Encoder self-attn + Decoder masked self-attn + Cross-attn | Encoder: bidirectional; Decoder: causal + cross | Map input → output | Translation, summarization | T5, original Transformer |

- Hugging Face course overview: [Transformer Architectures](https://huggingface.co/learn/llm-course/en/chapter1/6)

---

## 6) Why Masked Self-Attention Enables Generation

- Training objective: predict the next token given prior tokens.
- During training (teacher forcing), the model sees ground-truth prefixes and learns to produce the next token.
- At inference, it rolls forward by feeding its own predictions back in.
- The causal mask ensures the model cannot peek at future tokens, making it an honest generator.

---

## 7) Practical Implications and Workflows

- **Use encoder-only** when you need high-quality context-aware embeddings or understand an input (e.g., sentiment classification, semantic search, RAG embeddings).
- **Use decoder-only** when you need to generate text given a prompt (e.g., ChatGPT-style conversation, code generation, story writing).
- **Remember positional encoding** to respect word order. Without it, jumbled inputs can look the same to the model.

---

## 8) Summary

- Self-attention: can attend to both sides → powerful for understanding/encoding.
- Masked self-attention: can’t look ahead → necessary for generation.
- This subtle difference leads to very different model capabilities and use-cases.

---

## References

- Vaswani et al., 2017 — [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Jay Alammar — [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- Hugging Face — [Transformer Architectures](https://huggingface.co/learn/llm-course/en/chapter1/6)
- Mikolov et al., 2013 — [Efficient Estimation of Word Representations (Word2Vec)](https://arxiv.org/abs/1301.3781)
- Pennington et al., 2014 — [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf)
- Kazemnejad — [Transformer Positional Encoding](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/)
- Raschka — [Understanding and Coding Self-Attention](https://magazine.sebastianraschka.com/p/understanding-and-coding-self-attention)
- GeeksforGeeks — [How Do Self-Attention Masks Work?](https://www.geeksforgeeks.org/nlp/how-do-self-attention-masks-work/)
