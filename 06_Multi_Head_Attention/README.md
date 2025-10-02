<div style="background-color:#581C87; padding:20px; border-radius:10px; color:white;">
  <h1 align="center">Section 6: Multi-Head Attention</h1>
</div>
<br>

So far, we have a single, powerful attention mechanism. But what if one set of weights ($W_q, W_k, W_v$) isn't enough? A single attention mechanism might learn to focus on one type of relationship (e.g., how verbs relate to subjects), but miss out on other relationships (e.g., how adjectives modify nouns).

**Multi-Head Attention** solves this by running the entire attention mechanism multiple times in parallel, each with its own independent set of learnable weights. Each of these parallel runs is called an "attention head."

> **Analogy:** Imagine you're reading a dense report. You might read through it once to understand the main topic. Then, you might read it again to focus specifically on the financial numbers. Then, a third time to look for key dates. Each "read-through" is like an attention head—it's looking at the same text but from a different perspective.



## ⚙️ The Multi-Head Process

1.  **Initialization:** Instead of one set of weight matrices $(W_q, W_k, W_v)$, we initialize multiple sets. For example, for an 8-head attention, we would have $(W_q^0, W_k^0, W_v^0)$, $(W_q^1, W_k^1, W_v^1)$, ..., $(W_q^7, W_k^7, W_v^7)$.

2.  **Parallel Attention Calculation:** For each head $i$, we calculate the context vectors just like we did in the previous section:
    * $Q^i = X \cdot W_q^i$
    * $K^i = X \cdot W_k^i$
    * $V^i = X \cdot W_v^i$
    * `ContextVector^i = CausalAttention(Q^i, K^i, V^i)`

    This happens for all heads simultaneously. Each head produces its own context vector matrix.

3.  **Concatenation:** We take the output context vectors from all the heads and concatenate them together. If we have 8 heads and each produces a context vector of dimension 3, the concatenated vector for each token will have a dimension of $8 \times 3 = 24$.

    `ConcatenatedVector = [ContextVector^0, ContextVector^1, ..., ContextVector^7]`

4.  **Final Projection:** The concatenated vector is often too large and contains redundant information. So, we use one final learnable weight matrix, $W_o$ (for "output"), to project this large vector back down to the original input dimension (e.g., 3 in our case).

    `FinalOutput = ConcatenatedVector @ W_o`

This `FinalOutput` is the final result of the Multi-Head Attention layer. It's a context-rich vector that has incorporated information from multiple "perspectives" or "subspaces."

---
## 💡 Why This Works

Each attention head can learn to focus on different linguistic features.
* **Head 1** might focus on subject-verb agreement.
* **Head 2** might track pronoun references.
* **Head 3** might identify related concepts.
* **Head 4** might focus on the syntax of the sentence.

By combining the outputs, the model gets a much more comprehensive and nuanced understanding of the text.

### Dropout in Multi-Head Attention
Dropout can be applied after the softmax in each head and also to the final concatenated output before the last projection. This further helps in regularization.

### Attention Heads in Real Models
Real-world models use a large number of attention heads:
* **GPT-2 (small):** 12 heads
* **GPT-3:** 96 heads
* **LLaMA 2 (7B):** 32 heads

The combination of multiple heads and deep layers is what gives these models their incredible power.