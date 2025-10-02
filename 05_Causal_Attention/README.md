<div style="background-color:#047857; padding:20px; border-radius:10px; color:white;">
  <h1 align="center">Section 5: Causal Attention (Masking)</h1>
</div>
<br>

So far, our attention mechanism has allowed every word to look at every other word, including words that come *after* it. This is perfectly fine for tasks like text classification or sentiment analysis where the model sees the entire sentence at once.

However, for a **language model** whose job is to generate text one word at a time, this is a huge problem. It's like giving the model the answers to a test. If the model is trying to predict the next word after "studying," it should not have access to "artificial," "intelligence," or "engineering."

## 🎭 The Concept of Masking

To solve this, we introduce **causal attention**, also known as **masked self-attention**. The idea is simple: we "mask" out any information from future tokens before calculating the final attention weights.

We do this by adding a **mask matrix** to our attention scores, right before the softmax step. This mask matrix contains:
* `0` for all positions a token is allowed to attend to (itself and previous tokens).
* `-∞` (a very large negative number) for all future positions.

When you take the softmax of a number that includes `-∞`, the result becomes `0`. This effectively zeroes out the attention weight for any future tokens.

---

## 🧮 Masking in Action

Let's revisit the attention scores we calculated in the previous section:
$$
\text{Scores} =
\begin{bmatrix}
2.37 & 3.23 & 1.83 \\
3.23 & 4.38 & 2.49 \\
1.83 & 2.49 & 1.48
\end{bmatrix}
\quad
\begin{matrix}
\leftarrow \text{for "I"} \\
\leftarrow \text{for "am"} \\
\leftarrow \text{for "studying"}
\end{matrix}
$$

Our mask matrix for a 3-word sequence looks like this:
$$
\text{Mask} =
\begin{bmatrix}
0 & -\infty & -\infty \\
0 & 0 & -\infty \\
0 & 0 & 0
\end{bmatrix}
$$

Now, we add the mask to the scores:
`masked_scores = attention_scores + mask`
$$
\text{Masked Scores} =
\begin{bmatrix}
2.37 & -\infty & -\infty \\
3.23 & 4.38 & -\infty \\
1.83 & 2.49 & 1.48
\end{bmatrix}
$$

Finally, we scale and apply softmax as usual. Look at the resulting weights:

`attention_weights = softmax(masked_scores / sqrt(d_k))`

$$
\text{Causal Weights} =
\begin{bmatrix}
1.00 & 0.00 & 0.00 \\
0.22 & 0.78 & 0.00 \\
0.25 & 0.54 & 0.21
\end{bmatrix}
$$

**Interpretation:**
* **For "I" (row 0):** It can only attend to itself (100% weight).
* **For "am" (row 1):** It can attend to "I" (22%) and "am" (78%), but not "studying" (0%).
* **For "studying" (row 2):** It can attend to all three words.

This ensures that the output for any given token is only influenced by the past, preserving the causal nature required for text generation.

---

## 💧 What is Dropout?

In large neural networks, we often use a technique called **dropout** as a form of regularization to prevent **overfitting**. Overfitting is when a model learns the training data too well, including its noise, and performs poorly on new, unseen data.

Dropout works by randomly "dropping out" (setting to zero) a certain percentage of the neuron activations during training. In the context of attention, we can apply dropout to the attention weight matrix.

This forces the network to learn more robust features and prevents it from becoming too reliant on any single attention weight. It's like forcing a team to work together even if some members are randomly absent, making the whole team stronger.