<div style="background-color:#B45309; padding:20px; border-radius:10px; color:white;">
  <h1 align="center">Section 12: Finding the Next Word (Decoding Strategies)</h1>
</div>
<br>

We have a probability distribution over our entire vocabulary. The final step is to select a word from this distribution. This is called **decoding**. There are several strategies to do this, each with its own trade-offs between coherence, creativity, and correctness.

### 1. Greedy Decoding
This is the simplest strategy.
* **How it works:** Always choose the token with the highest probability.
* **Pros:** Very fast and computationally cheap.
* **Cons:** Extremely prone to being repetitive and producing boring, deterministic text. It might get stuck in loops (e.g., "I am I am I am..."). A high-probability word early on might lead to a dead-end sequence later.

### 2. Beam Search
An improvement over greedy search.
* **How it works:** At each step, keep track of the `k` (the "beam width") most probable sequences so far. For the next step, expand each of these `k` sequences with all possible next words and find the `k` new sequences with the highest overall probability.
* **Pros:** Produces more fluent and probable sequences than greedy search. Often used in machine translation.
* **Cons:** Can still be repetitive and may miss high-quality but lower-probability creative paths. It's also more computationally expensive.

### 3. Sampling
This introduces randomness.
* **How it works:** Instead of picking the most likely word, treat the probability distribution as a lottery and sample a word from it. A word with a 15% probability will be chosen 15% of the time.
* **Pros:** Generates much more diverse and creative text.
* **Cons:** Can be too random. There's a chance of picking nonsensical or very rare words, making the text incoherent.

### 4. Top-K Sampling
A way to control the randomness of sampling.
* **How it works:** First, filter the probability distribution to only include the `k` most probable tokens. Then, redistribute the probability mass among just these `k` tokens and sample from this smaller set. For example, if `k=50`, you only consider the 50 most likely words.
* **Pros:** A good balance. It prevents the model from picking absurdly unlikely words while still allowing for creativity.
* **Cons:** The number `k` is fixed. For a distribution where the probability is concentrated in a few words (a "peaked" distribution), `k=50` might be too large. For a flat distribution, `k=50` might be too small.

### 5. Top-p (Nucleus) Sampling 🏆
This is the state-of-the-art and most widely used method. It's an adaptive version of Top-K.
* **How it works:** Instead of picking a fixed number `k` of words, you pick the smallest set of words whose cumulative probability is greater than a certain threshold `p` (e.g., `p=0.95`). Then you sample from this set.
* **Example:** If `p=0.9`, you'd sort the words by probability and add them to your set one by one until their total probability exceeds 90%.
* **Why it's the best:** It's adaptive.
    * If the model is very confident about the next word (e.g., "artificial" has 92% probability), the nucleus will be very small, maybe just one word. The model acts like greedy search.
    * If the model is uncertain (many words have similar, low probabilities), the nucleus will be larger, allowing for more creative choices.
* This method is used in most modern chatbots, including those powered by GPT, as it provides the best balance of coherence and creativity.