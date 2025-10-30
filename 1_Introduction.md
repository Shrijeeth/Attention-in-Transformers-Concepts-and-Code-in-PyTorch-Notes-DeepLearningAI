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
