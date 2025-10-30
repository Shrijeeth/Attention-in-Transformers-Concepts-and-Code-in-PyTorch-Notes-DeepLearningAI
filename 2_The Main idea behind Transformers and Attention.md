# The Main idea behind Transformers and Attention

## Context: ChatGPT and Transformers

- **Hype and foundation**
- People are excited about ChatGPT, which is fundamentally built on the Transformer architecture.
- Transformers can look complex, but at their core rely on three parts: word embedding, positional encoding, and attention.

## The Three Fundamental Parts

- **Word embedding**
- **Positional encoding**
- **Attention**

## Word Embedding

- **Purpose**
- Converts tokens (words, subwords, symbols) into numbers.
- Needed because neural networks operate on numeric inputs only.

- **Example**
- Input: "tell me about pizza"
- The embedding layer maps each token to a numeric vector (an embedding) so the model can process it. Bam!

## Positional Encoding

- **Why it matters**
- Keeps track of word order so that meaning is preserved.

- **Example**
- "Squatch eats pizza" → sensible, response might be "yum."
- "Pizza eats Squatch" → very different meaning, response might be "Yikes!"

- **Key point**
- Same words, different order → different meaning.
- There are many ways to implement positional encoding; details are out of scope here.
- Just know positional encoding helps track order. Double bam!

## Attention (Self-Attention)

- **Goal**
- Establish relationships among words by measuring similarity between tokens.

- **Pronoun resolution example**
- Sentence: "the pizza came out of the oven and it tasted good"
- The word "it" could refer to "pizza" or to "oven".
- Attention helps the model correctly associate "it" with "pizza" (common in data), not "oven".

- **How self-attention works (conceptually)**
- For each word, compute similarity to all other words in the sentence, including itself.
- Do this for every word.
- Use these similarity scores to weight how each word is encoded, emphasizing more relevant words.

- **Effect**
- If "it" is more often associated with "pizza" in training data, the similarity score to "pizza" will be higher, so "pizza" has a larger impact on how "it" is encoded. Bam!

## Putting It All Together

- **Pipeline summary**
- Word embedding converts tokens to numbers.
- Positional encoding injects order information.
- Self-attention builds relationships among words to form context-aware representations.

- **Outcome**
- With these three components, a Transformer can represent meaning, respect word order, and relate words appropriately. Triple bam!

## Quick Glossary

- **Token**
- A basic unit of text (word, subword, or symbol) used as model input.

- **Embedding**
- A numeric vector representation of a token.

- **Positional encoding**
- Additional information that enables the model to account for token positions and order.

- **Self-attention**
- A mechanism where each token attends to all tokens (including itself) to compute context-aware representations.
