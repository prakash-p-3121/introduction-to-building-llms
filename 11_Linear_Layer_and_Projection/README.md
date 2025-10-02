<div style="background-color:#1D4ED8; padding:20px; border-radius:10px; color:white;">
  <h1 align="center">Section 11: Linear Layer & Projection to Vocabulary</h1>
</div>
<br>

After the input has passed through the entire stack of Transformer blocks, we are left with a final set of context-rich output vectors. Let's say for the last token in our input sequence ("engineering"), the final output vector `h` is:

`h = [2.5, -1.8, 0.9]` (This is our 3-dimensional example. In reality, it would be `d_model` dimensional, e.g., 768 or 4096).

This vector `h` contains a deep contextual understanding of the sequence "I am studying artificial intelligence engineering." But how do we use it to predict the *next* word?

We need to map this vector from the embedding space to the **vocabulary space**.

## 📏 The Final Linear Layer

The final step is to use a standard **Linear Layer** (also called a projection layer). This layer has a weight matrix `W` and a bias vector `b`.

The dimension of this weight matrix is `d_model` x `vocab_size`.
* `d_model`: The dimension of our output vector `h`.
* `vocab_size`: The total number of unique tokens in our vocabulary.

Let's say our vocabulary has 50,000 tokens. The weight matrix would be `3 x 50,000` in our tiny example, or `768 x 50,000` in a real model.

We compute the **logits** using the formula:
$$ \text{logits} = h \cdot W + b $$

The result, `logits`, is a very long vector with a size equal to our vocabulary size (50,000). Each element in the logits vector corresponds to a token in our vocabulary.

`logits = [1.2, -0.5, 3.4, ..., 0.8]` (A vector of length 50,000)

These are raw, unnormalized scores. A higher score means the model thinks that corresponding word is more likely to be the next word.

---

## 📊 The Softmax Function

The logits are useful, but they aren't probabilities. They don't sum to 1, and they can be positive or negative. To convert these scores into a proper probability distribution, we use the **Softmax** function one last time.

The softmax function is applied to the entire logits vector.

$$ \text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}} $$

Where $z$ is the logits vector and $V$ is the vocabulary size. This function exponentiates every logit (making them all positive) and then divides by the sum of all exponentiated logits.

The output is a **probability vector**, also of length 50,000, where:
* Every value is between 0 and 1.
* The sum of all values is exactly 1.

`probabilities = [0.01, 0.001, 0.15, ..., 0.009]`

Now, the value at index `i` of this vector represents the model's predicted probability that the token `i` from our vocabulary is the next word in the sequence. For example, if the token "and" is at index 2, the model is predicting a 15% chance that "and" is the next word.

We are now ready to choose the next word.