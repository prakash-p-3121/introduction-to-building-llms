<div style="background-color:#0F766E; padding:20px; border-radius:10px; color:white;">
  <h1 align="center">Section 1: Data Collection and Tokenization</h1>
</div>
<br>

Before an LLM can learn anything, it needs data—a lot of it. And that data needs to be converted into numbers. This section covers where the data comes from and how we turn raw text into a format our model can process.

## 🌐 Data Sources for Training LLMs

LLMs are trained on vast, internet-scale datasets. The goal is to expose the model to a diverse range of text, covering countless topics, styles, and languages. Common sources include:

* **Common Crawl:** An open repository of web crawl data containing trillions of words from the internet. It's massive but requires significant filtering to remove noise and low-quality content.
* **Wikipedia:** A high-quality, structured encyclopedia available in many languages. It's a reliable source for factual information.
* **Books:** Datasets like Google Books provide a wealth of long-form, well-written text that helps the model learn grammar and narrative structure.
* **GitHub:** For coding-focused LLMs, public code repositories on GitHub are invaluable for learning programming languages and logic.

The original **GPT-3** model by OpenAI was trained on a filtered version of Common Crawl, WebText2, Books1, Books2, and the English-language Wikipedia.

---

## 🗣️ From Text to Tokens: Tokenization

A model doesn't see words; it sees numbers. **Tokenization** is the process of breaking down a piece of text into smaller units, called **tokens**. A token can be a word, a part of a word, or even a single character.

Let's use our example sentence for the rest of this guide:
> **"I am studying artificial intelligence engineering"**

A simple tokenization strategy would be to split by space. This gives us the following tokens:
`["I", "am", "studying", "artificial", "intelligence", "engineering"]`

### Byte-Pair Encoding (BPE)
Modern LLMs often use more advanced tokenizers like **Byte-Pair Encoding (BPE)**. BPE starts with individual characters and iteratively merges the most frequently occurring pair of tokens. This allows it to handle any word, even those it hasn't seen before, by breaking them down into known sub-words. For example, "studying" might be tokenized as `["study", "ing"]`. For simplicity, we'll stick to our word-level tokens for now.

---

## 🔢 From Tokens to IDs to Vectors

Now we convert these tokens into numbers.

### 1. Token to Token ID
First, we build a vocabulary of all possible tokens and assign a unique integer ID to each.

| Token         | Token ID |
|---------------|----------|
| "I"           | 0        |
| "am"          | 1        |
| "studying"    | 2        |
| "artificial"  | 3        |
| "intelligence"| 4        |
| "engineering" | 5        |

Our sentence is now a sequence of integers: `[0, 1, 2, 3, 4, 5]`

### 2. Token ID to Vector Embedding
These IDs don't capture the meaning of the words. "artificial" (ID 3) isn't "3 times more important" than "am" (ID 1). We need a richer representation.

This is where **vector embeddings** come in. We map each token ID to a vector (a list of numbers). This vector represents the token's meaning in a high-dimensional space. Words with similar meanings will have similar vectors.

Let's use an **embedding dimension of 3** for our example. (Real models use dimensions like 768 or 4096).

| Token ID | Token         | Vector Embedding      |
|----------|---------------|-----------------------|
| 0        | "I"           | `[0.1, 0.4, 0.5]`     |
| 1        | "am"          | `[0.3, 0.8, 0.2]`     |
| 2        | "studying"    | `[0.9, 0.1, 0.8]`     |
| 3        | "artificial"  | `[0.2, 0.7, 0.6]`     |
| 4        | "intelligence"| `[0.4, 0.6, 0.7]`     |
| 5        | "engineering" | `[0.6, 0.5, 0.9]`     |

Our input is now a list of vectors! This is the actual input that goes into the first layer of our model.

---

## 📍 Positional Embeddings

There's a problem. The model processes all vectors at once. It has no idea about the order of the words. "I am studying" and "studying am I" would look the same!

We need to add information about the position of each word. We do this by creating a second vector called a **Positional Embedding** and adding it to the word embedding.

A common method uses a pair of sinusoidal functions:

![Positional Embedding Sine Formula](https://latex.codecogs.com/svg.latex?%5Cbg_white%20PE_%7B%28pos%2C%202i%29%7D%20%3D%20%5Csin%5Cleft%28%5Cfrac%7Bpos%7D%7B10000%5E%7B2i/d_%7B%5Ctext%7Bmodel%7D%7D%7D%7D%5Cright%29)

![Positional Embedding Cosine Formula](https://latex.codecogs.com/svg.latex?%5Cbg_white%20PE_%7B%28pos%2C%202i%2B1%29%7D%20%3D%20%5Ccos%5Cleft%28%5Cfrac%7Bpos%7D%7B10000%5E%7B2i/d_%7B%5Ctext%7Bmodel%7D%7D%7D%7D%5Cright%29)

- $pos$ is the position of the word (0, 1, 2, ...).
- $i$ is the index of the dimension in the embedding vector (0, 1, 2 for us).
- $d_{\text{model}}$ is the dimension of the embedding (3 for us).

Let's calculate the positional embedding for the word "studying" at `pos=2`:

- **Dimension 0 (i=0):** $PE_{(2, 0)} = \sin(2 / 10000^{0/3}) = \sin(2) \approx 0.909$
- **Dimension 1 (i=0):** $PE_{(2, 1)} = \cos(2 / 10000^{0/3}) = \cos(2) \approx -0.416$
- **Dimension 2 (i=1):** $PE_{(2, 2)} = \sin(2 / 10000^{2/3}) \approx \sin(0.0043) \approx 0.004$

So, the positional vector for "studying" is `[0.909, -0.416, 0.004]`.

**Final Input Vector for "studying":**
