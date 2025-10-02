<div style="background-color:#BE123C; padding:20px; border-radius:10px; color:white;">
  <h1 align="center">Section 4: Self-Attention with Trainable Weights</h1>
</div>
<br>

The simplified attention was great for understanding the process, but it had a major limitation: the way it measured "relevance" was fixed. The Query, Key, and Value were all the same thing.

To make the model truly powerful, we need to let it *learn* how to best transform our input vectors to calculate attention. We do this by introducing three **trainable weight matrices**:

* $W_q$: A matrix to transform input vectors into **Queries**.
* $W_k$: A matrix to transform input vectors into **Keys**.
* $W_v$: A matrix to transform input vectors into **Values**.

These matrices are initialized with random numbers and are updated during the model's training process via backpropagation. They allow the model to project the input embeddings into different "subspaces" that are optimal for calculating attention scores.

## ⚙️ The New Process

The process is almost the same, but with one extra step at the beginning.

Let $X$ be our input matrix of (word + positional) embeddings.

### Step 0: Project Inputs into Q, K, and V
We perform a matrix multiplication of our input $X$ with each of the new weight matrices.

1.  `Queries = X @ W_q`
2.  `Keys = X @ W_k`
3.  `Values = X @ W_v`

Let's assume we have our input matrix $X$ (first 3 words) and our randomly initialized weight matrices $W_q, W_k, W_v$. Our embeddings and weight matrices have a dimension of 3.

$$
X = \begin{bmatrix}
1.8 & 0.1 & 0.5 \\
0.3 & 1.2 & 0.8 \\
1.8 & -0.3 & 0.8
\end{bmatrix}
$$

$$
W_q = \begin{bmatrix} 0.1 & 0.2 & 0.3 \\ 0.4 & 0.5 & 0.6 \\ 0.7 & 0.8 & 0.9 \end{bmatrix},
\ W_k = \begin{bmatrix} 0.2 & 0.3 & 0.4 \\ 0.5 & 0.6 & 0.7 \\ 0.8 & 0.9 & 0.1 \end{bmatrix},
\ W_v = \begin{bmatrix} 0.3 & 0.4 & 0.5 \\ 0.6 & 0.7 & 0.8 \\ 0.9 & 0.1 & 0.2 \end{bmatrix}
$$

**Calculating Queries (Q):**
$$
Q = X \cdot W_q =
\begin{bmatrix}
0.57 & 0.81 & 0.99 \\
1.25 & 1.35 & 1.53 \\
0.62 & 0.45 & 0.63
\end{bmatrix}
$$

**Calculating Keys (K):**
$$
K = X \cdot W_k =
\begin{bmatrix}
0.81 & 1.08 & 1.24 \\
1.30 & 1.53 & 1.79 \\
0.85 & 0.87 & 0.59
\end{bmatrix}
$$

**Calculating Values (V):**
$$
V = X \cdot W_v =
\begin{bmatrix}
1.05 & 0.84 & 1.09 \\
1.47 & 1.04 & 1.15 \\
1.08 & 0.68 & 0.67
\end{bmatrix}
$$

---

## 🔁 The Same 3-Step Attention Calculation

Now that we have our learned Q, K, and V matrices, the rest of the process is exactly the same as before!

### Step 1: Calculate Attention Scores
`attention_scores = Q @ K.T`

$$
\text{Scores} =
\begin{bmatrix}
2.37 & 3.23 & 1.83 \\
3.23 & 4.38 & 2.49 \\
1.83 & 2.49 & 1.48
\end{bmatrix}
$$

### Step 2: Calculate Attention Weights (Softmax)
`attention_weights = softmax(attention_scores / sqrt(d_k))`

(After scaling by `sqrt(3)` and applying softmax...)
$$
\text{Weights} =
\begin{bmatrix}
0.17 & 0.65 & 0.18 \\
0.21 & 0.70 & 0.09 \\
0.25 & 0.54 & 0.21
\end{bmatrix}
$$

### Step 3: Calculate Context Vectors
`context_vectors = attention_weights @ V`

$$
\text{Context Vectors} =
\begin{bmatrix}
1.34 & 0.92 & 1.08 \\
1.37 & 0.95 & 1.09 \\
1.36 & 0.91 & 1.07
\end{bmatrix}
$$

This is the output of the self-attention layer! These new context vectors, which were computed using **learnable weights**, now have a much richer understanding of the relationships within the text.