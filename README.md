# Attention in Transformers: Concepts and Code in PyTorch

[![Course](https://img.shields.io/badge/DeepLearning.AI-Course-blue)](https://www.deeplearning.ai/short-courses/attention-in-transformers-concepts-and-code-in-pytorch/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9.0-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Comprehensive notes and PyTorch implementations from DeepLearning.AI's "Attention in Transformers" course taught by Josh Starmer (StatQuest).

## 📚 Overview

This repository contains detailed, beginner-friendly notes and clean PyTorch implementations covering the attention mechanism - the core component of modern transformer architectures that power large language models like GPT, BERT, and beyond.

### What's Inside

- **Detailed Markdown Notes**: Step-by-step explanations with visual aids, tables, and diagrams
- **PyTorch Implementations**: Clean, well-commented code for each attention variant
- **Jupyter Notebooks**: Interactive examples with shape validations and manual calculations
- **Mathematical Derivations**: Matrix math breakdowns with intuitive explanations

## 🗂️ Repository Structure

### Theory & Concepts

1. **[Introduction](1_Introduction.md)**
   - Course overview and motivation
   - Historical context (pre-transformer to modern LLMs)
   - Key terminology and timeline

2. **[Main Ideas Behind Transformers](2_The%20Main%20idea%20behind%20Transformers%20and%20Attention.md)**
   - Word embeddings and positional encoding
   - Core components overview
   - Self-attention intuition

3. **[Self-Attention Matrix Math](3_Self-Attention_Matrix_Math.md)**
   - Q, K, V derivation and database analogy
   - Scaled dot-product attention step-by-step
   - Shapes and dimensions cheat sheet

4. **[Self vs Masked Self-Attention](5_Self_vs_Masked_Self_Attention.md)**
   - Encoder-only vs decoder-only architectures
   - Causal masking explained
   - Use cases and practical implications

5. **[Masked Self-Attention Matrix Math](6_Masked_Self-Attention_Matrix_Math.md)**
   - Triangular mask construction
   - Look-ahead prevention mechanics
   - Example: "write a poem" walkthrough

6. **[Encoder-Decoder Attention (Cross-Attention)](8_Encoder_Decoder_Attention_and_Cross_Attention.md)**
   - Where the names come from
   - Cross-attention mechanics (Q from decoder, K/V from encoder)
   - Translation example and multimodal applications

7. **[Multi-Head Attention](9_Multi-Head_Attention.md)**
   - Why multiple heads help
   - Parallel attention computation
   - Concatenation and output projection

### Code Implementations

1. **[Coding Self-Attention](4_Coding_Self_Attention_in_Pytorch.ipynb)**
   - `SelfAttention` class implementation
   - Manual weight inspection and validation
   - Shape verification

2. **[Coding Masked Self-Attention](7_Coding_Masked_Self_Attention_in_Pytorch.ipynb)**
   - `MaskedSelfAttention` with causal mask
   - `torch.tril` mask construction
   - Step-by-step computation verification

3. **[Coding Encoder-Decoder & Multi-Head Attention](10_Coding_Encoder-Decoder_and_Multi_Head_Attention.ipynb)**
   - Generic `Attention` class (self/cross)
   - `MultiHeadAttention` wrapper
   - Concatenation and shape handling

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
PyTorch 2.9.0
```

### Installation

1. Clone the repository:

```bash
git clone https://github.com/[your-username]/Attention-in-Transformers-Concepts-and-Code-in-PyTorch-Notes-DeepLearningAI.git
cd Attention-in-Transformers-Concepts-and-Code-in-PyTorch-Notes-DeepLearningAI
```

1. Install dependencies:

```bash
pip install -r requirements.txt
```

1. Launch Jupyter notebooks:

```bash
jupyter notebook
```

## 📖 Key Concepts Covered

### Attention Mechanisms

- **Self-Attention**: Bidirectional context (encoder-only models like BERT)
- **Masked Self-Attention**: Causal/autoregressive (decoder-only models like GPT)
- **Cross-Attention**: Encoder-decoder connection (translation, multimodal)
- **Multi-Head Attention**: Parallel attention in different subspaces

### Mathematical Components

- Query (Q), Key (K), Value (V) projections
- Scaled dot-product: `Attention(Q,K,V) = softmax(QK^T/√d_k)V`
- Positional encoding (sine/cosine)
- Masking strategies (padding, causal)

### Architecture Patterns

- **Encoder-only**: Understanding tasks (classification, embeddings)
- **Decoder-only**: Generation tasks (text completion, chat)
- **Encoder-Decoder**: Seq2seq tasks (translation, summarization)

## 🎯 Learning Path

**For Beginners:**

1. Start with `1_Introduction.md` for context
2. Read `2_The Main idea behind Transformers and Attention.md` for intuition
3. Work through `3_Self-Attention_Matrix_Math.md` with paper and pen
4. Run `4_Coding_Self_Attention_in_Pytorch.ipynb` to see it in action

**For Practitioners:**

1. Jump to specific attention variants (masked, cross, multi-head)
2. Study the PyTorch implementations for production insights
3. Compare encoder-only vs decoder-only trade-offs in `5_Self_vs_Masked_Self_Attention.md`

## 🔗 Resources

- **Course**: [DeepLearning.AI - Attention in Transformers](https://www.deeplearning.ai/short-courses/attention-in-transformers-concepts-and-code-in-pytorch/)
- **Instructor**: Josh Starmer ([StatQuest](https://statquest.org/))
- **Paper**: [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- **Visual Guide**: [The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/)
- **PyTorch Docs**: [nn.MultiheadAttention](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)

## 🤝 Contributing

Found a typo or want to improve explanations? Contributions are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -m 'Add improvement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **DeepLearning.AI** for the excellent course structure
- **Josh Starmer** for making complex concepts accessible
- **Vaswani et al.** for the original Transformer paper
- **Jay Alammar** for visual explanations that inspired the note-taking style

⭐ If you found these notes helpful, please consider starring the repository!

**Note**: These are educational notes created while learning. For production use, refer to official PyTorch implementations and the original research papers.
