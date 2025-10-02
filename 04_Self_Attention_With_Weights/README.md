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

![Initial X Matrix](https://latex.codecogs.com/svg.latex?%5Cbg_white%20X%20%3D%20%5Cbegin%7Bbmatrix%7D%201.8%20%26%200.1%20%26%200.5%20%5C%5C%200.3%20%26%201.2%20%26%200.8%20%5C%5C%201.8%20%26%20-0.3%20%26%200.8%20%5Cend%7Bbmatrix%7D)

![Weight Matrices Wq, Wk, Wv](https://latex.codecogs.com/svg.latex?%5Cbg_white%20W_q%20%3D%20%5Cbegin%7Bbmatrix%7D%200.1%20%26%200.2%20%26%200.3%20%5C%5C%200.4%20%26%200.5%20%26%200.6%20%5C%5C%200.7%20%26%200.8%20%26%200.9%20%5Cend%7Bbmatrix%7D%2C%20%5C%3B%20W_k%20%3D%20%5Cbegin%7Bbmatrix%7D%200.2%20%26%200.3%20%26%200.4%20%5C%5C%200.5%20%26%200.6%20%26%200.7%20%5C%5C%200.8%20%26%200.9%20%26%200.1%20%5Cend%7Bbmatrix%7D%2C%20%5C%3B%20W_v%20%3D%20%5Cbegin%7Bbmatrix%7D%200.3%20%26%200.4%20%26%200.5%20%5C%5C%200.6%20%26%200.7%20%26%200.8%20%5C%5C%200.9%20%26%200.1%20%26%200.2%20%5Cend%7Bbmatrix%7D)

**Calculating Queries (Q):**
![Q Calculation](https://latex.codecogs.com/svg.latex?%5Cbg_white%20Q%20%3D%20X%20%5Ccdot%20W_q%20%3D%20%5Cbegin%7Bbmatrix%7D%200.57%20%26%200.81%20%26%200.99%20%5C%5C%201.25%20%26%201.35%20%26%201.53%20%5C%5C%200.62%20%26%200.45%20%26%200.63%20%5Cend%7Bbmatrix%7D)

**Calculating Keys (K):**
![K Calculation](https://latex.codecogs.com/svg.latex?%5Cbg_white%20K%20%3D%20X%20%5Ccdot%20W_k%20%3D%20%5Cbegin%7Bbmatrix%7D%200.81%20%26%201.08%20%26%201.24%20%5C%5C%201.30%20%26%201.53%20%26%201.79%20%5C%5C%200.85%20%26%200.87%20%26%200.59%20%5Cend%7Bbmatrix%7D)

**Calculating Values (V):**
![V Calculation](https://latex.codecogs.com/svg.latex?%5Cbg_white%20V%20%3D%20X%20%5Ccdot%20W_v%20%3D%20%5Cbegin%7Bbmatrix%7D%201.05%20%26%200.84%20%26%201.09%20%5C%5C%201.47%20%26%201.04%20%26%201.15%20%5C%5C%201.08%20%26%200.68%20%26%200.67%20%5Cend%7Bbmatrix%7D)

---

## 🔁 The Same 3-Step Attention Calculation

Now that we have our learned Q, K, and V matrices, the rest of the process is exactly the same as before!

### Step 1: Calculate Attention Scores
`attention_scores = Q @ K.T`
![Attention Scores with Weights](https://latex.codecogs.com/svg.latex?%5Cbg_white%20%5Ctext%7BScores%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%202.37%20%26%203.23%20%26%201.83%20%5C%5C%203.23%20%26%204.38%20%26%202.49%20%5C%5C%201.83%20%26%202.49%20%26%201.48%20%5Cend%7Bbmatrix%7D)


### Step 2: Calculate Attention Weights (Softmax)
`attention_weights = softmax(attention_scores / sqrt(d_k))`
![Attention Weights with Weights](https://latex.codecogs.com/svg.latex?%5Cbg_white%20%5Ctext%7BWeights%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%200.17%20%26%200.65%20%26%200.18%20%5C%5C%200.21%20%26%200.70%20%26%200.09%20%5C%5C%200.25%20%26%200.54%20%26%200.21%20%5Cend%7Bbmatrix%7D)

### Step 3: Calculate Context Vectors
`context_vectors = attention_weights @ V`
![Context Vectors with Weights](https://latex.codecogs.com/svg.latex?%5Cbg_white%20%5Ctext%7BContext%20Vectors%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%201.34%20%26%200.92%20%26%201.08%20%5C%5C%201.37%20%26%200.95%20%26%201.09%20%5C%5C%201.36%20%26%200.91%20%26%201.07%20%5Cend%7Bbmatrix%7D)

This is the output of the self-attention layer! These new context vectors, which were computed using **learnable weights**, now have a much richer understanding of the relationships within the text.
