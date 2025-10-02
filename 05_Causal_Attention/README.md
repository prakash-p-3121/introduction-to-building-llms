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
![Attention Scores to be Masked](https://latex.codecogs.com/svg.latex?%5Cbg_white%20%5Ctext%7BScores%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%202.37%20%26%203.23%20%26%201.83%20%5C%5C%203.23%20%26%204.38%20%26%202.49%20%5C%5C%201.83%20%26%202.49%20%26%201.48%20%5Cend%7Bbmatrix%7D)

Our mask matrix for a 3-word sequence looks like this:
![Mask Matrix](https://latex.codecogs.com/svg.latex?%5Cbg_white%20%5Ctext%7BMask%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%200%20%26%20-%5Cinfty%20%26%20-%5Cinfty%20%5C%5C%200%20%26%200%20%26%20-%5Cinfty%20%5C%5C%200%20%26%200%20%26%200%20%5Cend%7Bbmatrix%7D)

Now, we add the mask to the scores:
`masked_scores = attention_scores + mask`
![Masked Scores](https://latex.codecogs.com/svg.latex?%5Cbg_white%20%5Ctext%7BMasked%20Scores%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%202.37%20%26%20-%5Cinfty%20%26%20-%5Cinfty%20%5C%5C%203.23%20%26%204.38%20%26%20-%5Cinfty%20%5C%5C%201.83%20%26%202.49%20%26%201.48%20%5Cend%7Bbmatrix%7D)

Finally, we scale and apply softmax as usual. Look at the resulting weights:

`attention_weights = softmax(masked_scores / sqrt(d_k))`
![Causal Attention Weights](https://latex.codecogs.com/svg.latex?%5Cbg_white%20%5Ctext%7BCausal%20Weights%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%201.00%20%26%200.00%20%26%200.00%20%5C%5C%200.22%20%26%200.78%20%26%200.00%20%5C%5C%200.25%20%26%200.54%20%26%200.21%20%5Cend%7Bbmatrix%7D)

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
