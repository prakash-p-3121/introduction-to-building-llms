<div style="background-color:#9333EA; padding:20px; border-radius:10px; color:white;">
  <h1 align="center">Section 3: Simplified Self-Attention</h1>
</div>
<br>

Let's build the simplest version of self-attention to understand the core mechanics. In this version, we don't have any learnable weights yet. We'll use the (word + positional) embedding vectors directly.

Our goal is to produce a new vector for each word, called a **context vector**, that is a blend of all other word vectors in the sentence, weighted by their relevance.

## 🤝 Measuring Relevance: Dot Product & Cosine Similarity

How do we determine if two words are "relevant" to each other? We use their vectors! If two vectors point in a similar direction in the embedding space, the words they represent are likely related.

A simple way to measure this similarity is the **dot product**. A large dot product between two vectors means they are highly aligned.

> **Dot Product:** For two vectors `a = [a1, a2]` and `b = [b1, b2]`, the dot product is `a · b = a1*b1 + a2*b2`.

---

## 🧐 Queries, Keys, and Values (Q, K, V)

The attention mechanism is often explained using the concepts of **Queries**, **Keys**, and **Values**.

* **Query (Q):** The vector for the current word we are focusing on. It's asking the question: "Who in this sentence is relevant to me?"
* **Key (K):** The vectors for all words in the sentence (including the current one). The query is compared against each key to calculate relevance scores.
* **Value (V):** The vectors for all words in the sentence. These are the vectors we will blend together to get our final output.

In this simplified version: **Query = Key = Value = Our initial input vectors.**

Let's use just the first three words for this example to keep the matrices small. We'll use the combined (word + positional) embeddings.
`X = [x_I, x_am, x_studying]`

![Equation for X](https://latex.codecogs.com/svg.latex?%5Cbg_white%20X%20%3D%20%5Cbegin%7Bbmatrix%7D%201.8%20%26%200.1%20%26%200.5%20%5C%5C%200.3%20%26%201.2%20%26%200.8%20%5C%5C%201.8%20%26%20-0.3%20%26%200.8%20%5Cend%7Bbmatrix%7D%20%5Cquad%20%5Cbegin%7Bmatrix%7D%20%5Cleftarrow%20%5Ctext%7BI%7D%20%5C%5C%20%5Cleftarrow%20%5Ctext%7Bam%7D%20%5C%5C%20%5Cleftarrow%20%5Ctext%7Bstudying%7D%20%5Cend%7Bmatrix%7D)

So, `Queries = Keys = Values = X`.

---

## 🧮 The 3-Step Attention Calculation

### Step 1: Calculate Attention Scores
We calculate the dot product of each word's **Query** with every other word's **Key**.

`attention_scores = Queries @ Keys.T` (where `.T` is Transpose)

$$
\begin{bmatrix}
1.8 & 0.1 & 0.5 \\
0.3 & 1.2 & 0.8 \\
1.8 & -0.3 & 0.8
\end{bmatrix}
\times
\begin{bmatrix}
1.8 & 0.3 & 1.8 \\
0.1 & 1.2 & -0.3 \\
0.5 & 0.8 & 0.8
\end{bmatrix}
=
\begin{bmatrix}
3.50 & 1.06 & 3.61 \\
1.06 & 1.70 & 1.12 \\
3.61 & 1.12 & 3.97
\end{bmatrix}
$$

**Interpretation:** The score in `[row 2, col 0]` is `1.06`. This is the relevance score of word "I" (col 0) to word "am" (row 2). The highest score is `3.97`, the relevance of "studying" to itself.

### Step 2: Calculate Attention Weights (Softmax)
The scores are hard to interpret. We convert them into probabilities (that sum to 1) using the **softmax** function. We also scale them by dividing by the square root of the key dimension ($d_k=3$) to stabilize training.

`attention_weights = softmax(attention_scores / sqrt(d_k))`

First, scale by `sqrt(3) ≈ 1.732`:
$$
\text{Scaled Scores} = \begin{bmatrix}
2.02 & 0.61 & 2.08 \\
0.61 & 0.98 & 0.65 \\
2.08 & 0.65 & 2.30
\end{bmatrix}
$$
Now, apply softmax to each row:
$$
\text{Attention Weights} = \begin{bmatrix}
0.44 & 0.10 & 0.46 \\
0.25 & 0.40 & 0.35 \\
0.45 & 0.18 & 0.37
\end{bmatrix}
$$

**Interpretation:** Look at the row for "studying" (`[0.45, 0.18, 0.37]`). This means to calculate the new vector for "studying," we should take 45% from "I," 18% from "am," and 37% from "studying" itself.

### Step 3: Calculate Context Vectors
Finally, we create the context vectors by multiplying the **attention weights** with the **Value** matrix.

`context_vectors = attention_weights @ Values`

$$
\begin{bmatrix}
0.44 & 0.10 & 0.46 \\
0.25 & 0.40 & 0.35 \\
0.45 & 0.18 & 0.37
\end{bmatrix}
\times
\begin{bmatrix}
1.8 & 0.1 & 0.5 \\
0.3 & 1.2 & 0.8 \\
1.8 & -0.3 & 0.8
\end{bmatrix}
$$

![Context Vector Result](https://latex.codecogs.com/svg.latex?%5Cbg_white%20%5Cbegin%7Bbmatrix%7D%201.64%20%26%200.02%20%26%200.67%20%5C%5C%201.20%20%26%200.39%20%26%200.83%20%5C%5C%201.53%20%26%20-0.02%20%26%200.60%20%5Cend%7Bbmatrix%7D%20%5Cquad%20%5Cbegin%7Bmatrix%7D%20%5Cleftarrow%20%5Ctext%7BContext%20for%20%22I%22%7D%20%5C%5C%20%5Cleftarrow%20%5Ctext%7BContext%20for%20%22am%22%7D%20%5C%5C%20%5Cleftarrow%20%5Ctext%7BContext%20for%20%22studying%22%7D%20%5Cend%7Bmatrix%7D)

<br>

We now have new, context-rich vectors! The vector for "studying" `[1.53, -0.02, 0.60]` is a weighted blend of all input vectors, informed by their relevance. This output is then passed to the next part of the Transformer block.
