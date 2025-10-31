# Introduction

## Course Context and Instructors

- **Instructor**
  Josh Starmer (CEO, StatQuest), teaching concepts and PyTorch implementations.
- **Introduction**
  Andrew provides historical context and motivation; Josh covers algorithms and code with illustrations.

## What You Will Learn

- **Attention Mechanism**
  The key breakthrough that enabled Transformers.
- **Evolution and Implementation**
  How attention developed, how it works, and how to implement it.
- **Transformer Architecture**
  Why it's crucial for modern LLMs and how it scales on GPUs.
- **Hands-on PyTorch**
  Matrix math and coding of attention, including self-attention variants.
- **Architectural Patterns**
  Encoder-decoder, multi-head attention, and practical usage.

## Motivation: Machine Translation Challenges (Pre-Transformers)

- **Naive Word Mapping Fails**
  Word-by-word lookup ignores context, grammar, and syntax.
- **Word Order Differences**
  Example: "the European Economic Area was…" reordered in French.
- **Length Mismatch**
  Example: "They arrived late" (3 words) → 5 words in French.
- **Need for Context**
  Translation depends on surrounding words and sentence-level meaning.

## Early Attention (circa 2014)

- **Pioneering Groups**
  Yoshua Bengio's group (Université de Montréal) and Chris Manning's group (Stanford) independently proposed attention.
- **Encoder-Decoder with Attention**
  Encoder processes the input sequence; decoder generates the output sequence.
- **From Single Vector to Per-Word Vectors**
  Earlier models compressed a sentence into one dense vector. Newer models preserved one vector per input word.
- **Contextual Embeddings**
  Each word's vector represents its meaning in context of the sentence.
- **Decoder Attends to Inputs**
  At each step, the decoder weights (attends to) encoder outputs differently based on the output position and source positions.
- **Illustrative Behavior**
  First French word may attend to the first English word; second French word may attend to "area" (fourth English word) due to reordering.

## Mechanism: Encoder-Decoder with Attention (High-Level)

- **Encoder**
  Reads tokens one at a time, producing hidden vectors per token (context-aware).
- **Attention Weights**
  For each output step, compute weights over all encoder vectors to focus on the most relevant source tokens.
- **Context Vector**
  Weighted combination of encoder outputs guides the decoder's next word prediction.
- **Autoregressive Decoding**
  Decoder generates one token at a time, conditioned on prior outputs.

## Transformers (2017): "Attention Is All You Need"

- **Authors and Origin**
  Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin; from Google Brain.
- **Core Idea**
  A general, highly scalable attention-based architecture designed for GPUs.
- **Architecture Split**
  Encoder creates contextual embeddings in a single pass; decoder generates outputs step-by-step, feeding previous outputs back in as context.
- **Design Priority**
  Scalability on GPUs drove key architectural choices.
- **Impact on Modern Models**
  Foundations for today's LLMs and embedding models.

## Model Lineage and Applications

- **Encoder-Only: BERT**
  "Bidirectional Encoder Representations from Transformers."
  Basis for most embedding models used for RAG and recommender systems.
- **Decoder-Only: GPT**
  "Generative Pre-trained Transformer."
  Basis for ChatGPT and models from OpenAI and others (Anthropic, Google, Mistral, Meta).
- **Scaling Depth**
  Original paper: 6 layers of attention.
  Modern large models: e.g., Llama 3.2-405B uses 126 layers while keeping the same basic architecture.

## Course Roadmap

- **Foundations**
  Main ideas behind Transformers and Attention.
- **Math and Code**
  Matrix operations and PyTorch implementation of attention.
- **Attention Variants**
  Self-attention vs. masked self-attention; practical PyTorch coding.
- **Architectures**
  Encoder-decoder and multi-head attention, with step-by-step builds.

## Key Terms (Concise)

- **Attention**
  A mechanism that computes weights over inputs to focus on the most relevant information for the current step.
- **Encoder**
  Transforms input tokens into contextual embeddings (one vector per token).
- **Decoder**
  Autoregressively generates outputs, attending to encoder outputs (and prior outputs).
- **Contextual Embedding**
  A token representation that depends on surrounding tokens (context).
- **Self-Attention**
  Tokens attend to other tokens in the same sequence (used in both encoder and decoder).
- **Masked Self-Attention**
  Decoder variant where tokens cannot attend to future positions (prevents "cheating" during generation).
- **Cross-Attention**
  Decoder attends to encoder outputs (source-to-target attention in encoder-decoder models).

## Historical Timeline

- **2014**
  Attention introduced for neural machine translation by groups at Montréal and Stanford.
  Move from single sentence vector to per-word contextual embeddings with attention.
- **2017**
  Transformer architecture ("Attention Is All You Need"): scalable, encoder-decoder with attention, basis for modern LLMs.

## Practical Implications

- **LLMs and Embeddings**
  Attention underpins GPT-style generators and BERT-like encoders used in RAG and recommendations.
- **Scalability**
  Transformer's design aligns with GPU acceleration, enabling deep, large models.
- **Generalization**
  The same attention principles apply beyond translation to many sequence tasks.

## Beginner's Guide: Transformers and Attention

A gentle, visual-first overview of what Transformers are and how attention works, with references for further reading.

### What is a Transformer?

The Transformer is a neural network architecture introduced in the 2017 paper “Attention Is All You Need” that processes sequences (like text) efficiently by using attention instead of recurrence or convolutions.

![Transformer architecture diagram (Wikimedia Commons)](https://commons.wikimedia.org/wiki/Special:FilePath/Transformer%2C_full_architecture.png)

- Source: [Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762) — arXiv
- Visual walkthrough: [The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/)

### Transformers at a Glance

| Component | What it does | Why it matters | Simple intuition |
| --- | --- | --- | --- |
| Word Embeddings | Convert tokens (words/subwords) into numeric vectors | Neural nets need numbers | “Lookup” a dense vector for each token |
| Positional Encoding | Inject order information into embeddings | Word order changes meaning | Add patterns (e.g., sines/cosines) so the model knows positions |
| Self-Attention | Lets each token “look at” other tokens | Captures relationships and context | Weighted averaging of other tokens’ information |
| Feedforward Layers | Nonlinear transformation per token | Adds capacity beyond attention | Small MLP applied to each position |
| Layer Norm + Residuals | Stabilize and ease training | Enable deeper networks | Keep signals well-scaled and trainable |

### Types of Transformer Models

| Model type | Context window | Typical tasks | Examples |
| --- | --- | --- | --- |
| Encoder-only | Bidirectional (looks left and right) | Understanding: classification, NER, sentence embeddings | BERT, RoBERTa |
| Decoder-only | Left-to-right (causal, masked) | Generation: next-token prediction, chat, coding | GPT, Llama, Mistral |
| Encoder–Decoder | Encoder understands input; decoder generates output using cross-attention | Seq2Seq: translation, summarization | T5, original Transformer |

Further reading: [Hugging Face LLM Course — Architectures](https://huggingface.co/learn/llm-course/en/chapter1/6)

### The Three Building Blocks (Beginner Version)

1. Word Embedding: Maps each token to a vector of numbers. Example: “tell me about pizza” → vectors per token.
2. Positional Encoding: Adds position info so “Squatch eats pizza” ≠ “Pizza eats Squatch”. Sinusoidal PE explainer: [Positional Encoding (sine/cosine) explained](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/)
3. Self-Attention (core idea): For each token, compute how related it is to every other token and use those relations as weights to mix others’ information into a new, context-aware representation. [Interactive demo (GPT‑2 attention)](https://poloclub.github.io/transformer-explainer/)

### Self-Attention: Step by Step (Scaled Dot-Product)

Given encoded tokens X (embeddings + positions):

1. Create Queries, Keys, Values: Q = X·W_Q, K = X·W_K, V = X·W_V (learned weight matrices)
2. Similarity scores: S = Q·K^T (dot products measure how related tokens are)
3. Scale: S_scaled = S / sqrt(d_k) (helps keep values numerically stable)
4. Row-wise Softmax: A = softmax_rows(S_scaled) (turn each row into percentages that sum to 1)
5. Weighted sum of values: O = A·V (new context-aware representations)

Links:

- [Attention Is All You Need (arXiv)](https://arxiv.org/abs/1706.03762)
- [PyTorch MultiheadAttention docs](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)
- [The Illustrated Transformer (visual guide)](https://jalammar.github.io/illustrated-transformer/)

### Key Symbols and Terms

| Symbol/Term | Meaning | Notes |
| --- | --- | --- |
| Q (Query) | What a token uses to look for relevant info | “What am I looking for?” |
| K (Key) | What other tokens expose for being found | “How others can be found” |
| V (Value) | The information carried by tokens | “What to take if a match is found” |
| d_model | Embedding size per token | Often 256–4096+ |
| d_k, d_v | Sizes for Q/K and V projections | Often d_model/num_heads |
| softmax | Turns numbers into probabilities per row | Each row sums to 1 |

### Practical Tips for Beginners

- Start with encoder-only (BERT) for classification/embeddings, decoder-only (GPT) for generation.
- Use pretrained checkpoints; fine-tune for your task.
- Keep batch/sequence lengths moderate at first to control memory.
- Inspect attention maps with interactive tools to build intuition.

### More Resources

- [Attention Is All You Need (paper)](https://arxiv.org/abs/1706.03762)
- [BERT (paper)](https://arxiv.org/abs/1810.04805)
- [Hugging Face course (Transformers)](https://huggingface.co/learn/llm-course)
- [UvA DL Tutorial: Transformers and Multi-Head Attention](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html)
- [Karpathy “Zero to Hero”](https://karpathy.ai/zero-to-hero.html)
