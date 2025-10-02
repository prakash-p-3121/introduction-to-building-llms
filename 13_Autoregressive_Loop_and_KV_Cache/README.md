<div style="background-color:#3730A3; padding:20px; border-radius:10px; color:white;">
  <h1 align="center">Section 13: Autoregressive Loop & KV Caching</h1>
</div>
<br>

We've successfully predicted one word! But language models generate long sequences of text. How do they do it? Through a process called the **autoregressive loop**.

## 🔄 The Autoregressive Loop

"Autoregressive" means that the model's prediction at the current step is fed back as an input for the next step.

Here's the loop for generating text:

1.  **Start:** You provide an initial prompt, e.g., "I am studying".
2.  **Process:** The model processes this prompt and predicts the next token, e.g., "artificial".
3.  **Append:** The model appends the predicted token to the input sequence. The new input is now "I am studying artificial".
4.  **Repeat:** The model takes this new, longer sequence as its input and processes it from scratch to predict the *next* token (e.g., "intelligence").
5.  **Continue:** This loop continues, appending one word at a time, until the model generates a special `[END_OF_SEQUENCE]` token or reaches a predefined maximum length.

### The Inefficiency Problem
There's a massive inefficiency here. In step 4, when predicting the word after "artificial", the model re-calculates the attention for "I", "am", and "studying" all over again. As the sequence gets longer, the computation required at each step grows quadratically, making generation very slow.

## ⚡ The Solution: Key-Value (KV) Caching

This is where a crucial optimization called **KV Caching** comes in.

Recall that during self-attention, we create Query (Q), Key (K), and Value (V) matrices. The K and V matrices for a given token, once computed, **do not change** in subsequent generation steps.

KV Caching takes advantage of this.

**How it works:**

1.  **Initial Prompt ("I am studying"):**
    * The model processes the prompt.
    * For each layer and each attention head, it computes the Key and Value matrices for the tokens "I", "am", and "studying".
    * Instead of throwing them away, it **stores (caches) these K and V matrices**.
    * It then computes the first new token, "artificial".

2.  **Next Step (Generating after "artificial"):**
    * The input to the model is now just the *single new token*, "artificial".
    * For each layer and head, the model computes the Q, K, and V vectors *only for this new token*.
    * It then retrieves the cached K and V matrices from the previous step.
    * It **concatenates** the new K and V vectors to the cached ones.
    * It performs the attention calculation using the query from "artificial" and the full, concatenated K and V matrices.
    * Finally, it updates the cache by appending the new K and V vectors to it.

**Why is this a huge deal?**
At each step, we only need to perform the Transformer calculations for a single token. We avoid re-computing the attention for the entire context every time. This changes the complexity from being quadratic with the sequence length to being linear, making text generation significantly faster and feasible for long documents. KV caching is a critical engineering optimization for making LLMs practical to use.

