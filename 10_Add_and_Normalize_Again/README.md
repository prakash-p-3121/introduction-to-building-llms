<div style="background-color:#44403C; padding:20px; border-radius:10px; color:white;">
  <h1 align="center">Section 10: Add & Normalize Again</h1>
</div>
<br>

We've reached the final step inside a Transformer block! This step is identical in function to the first "Add & Norm" layer we saw in Section 8, but it's applied to the output of the Feed-Forward Network (FFN) sub-layer.

## ✅ The Final Block Procedure

Let's recap the flow for a single token's vector through the second half of the block:

1.  **Input:** The vector comes from the output of the *first* "Add & Norm" layer. Let's call this `input_to_ffn`.
    `input_to_ffn = [1.23, -1.22, -0.01]` (from our example in Section 8)

2.  **FFN Processing:** This vector is passed through the Feed-Forward Network.
    `ffn_output = FFN(input_to_ffn)`
    Let's imagine the FFN processes it and produces a new vector:
    `ffn_output = [0.95, -1.50, 0.25]`

3.  **Add (Residual Connection):** We add the input of the FFN layer to its output.
    `residual_output = input_to_ffn + ffn_output`
    ```
      [1.23, -1.22, -0.01]  (Input to FFN)
    + [0.95, -1.50, 0.25]  (FFN Output)
    --------------------------
    = [2.18, -2.72, 0.24]  (Residual Output)
    ```

4.  **Norm (Layer Normalization):** We normalize this resulting vector to have a mean of 0 and a variance of 1, and then apply the learnable scale ($\gamma$) and shift ($\beta$) parameters. This produces the final output vector for this Transformer block.

---

## 🏁 End of the Transformer Block

**This is the end of one complete Transformer block.**

The output is a set of vectors, one for each token, that are now significantly more context-aware than when they started.

* The **Multi-Head Attention** step allowed tokens to exchange information and understand their relationships.
* The **Feed-Forward Network** step allowed the model to process this new information for each token individually.
* The **Add & Norm** layers ensured the whole process was stable and that information from previous layers was preserved.

### ➡️ What's Next?

This final output vector is then passed as the input to the **next Transformer block**. The entire process we've just described repeats itself, with the model building up a more and more sophisticated understanding of the text with each subsequent block.

After passing through the final Transformer block in the stack (e.g., the 96th block in GPT-3), we have our final, context-rich vector representations. The next step is to use these vectors to actually predict the next word.