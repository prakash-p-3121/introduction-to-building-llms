<div style="background-color:#9A3412; padding:20px; border-radius:10px; color:white;">
  <h1 align="center">Section 8: Add & Normalize Layer</h1>
</div>
<br>

After the Multi-Head Attention sub-layer (and after the Feed-Forward sub-layer), we have a crucial step called "Add & Norm." This consists of two parts: a residual connection (Add) and layer normalization (Norm).

## ➕ Part 1: The "Add" - Residual Connections

A **residual connection** is a simple but powerful idea: we add the input of a layer to its output.

`output = Input + SubLayer(Input)`

Let's say the input vector for "studying" going into our Multi-Head Attention layer was:
`input = [1.8, -0.3, 0.8]`

And after passing through the Multi-Head Attention, the resulting context vector was:
`attention_output = [1.36, 0.91, 1.07]`

The residual connection simply adds them together element-wise:
```
  [1.8, -0.3, 0.8]   (Input)
+ [1.36, 0.91, 1.07]  (Attention Output)
-----------------------
= [3.16, 0.61, 1.87]   (Residual Output)
```

### Why do we do this?
As we stack many layers (dozens or even hundreds), it can become hard for the model to learn. Gradients can vanish or explode during backpropagation, making training impossible.

Residual connections create a "shortcut" for the gradient to flow through the network. It ensures that even if a sub-layer (like attention) temporarily messes up and learns nothing useful, the output is at least as good as the input. This makes training very deep networks much more stable.

---

## 📏 Part 2: The "Norm" - Layer Normalization

After the addition, the values in our vector might become very large or very small. This can make training unstable. **Layer Normalization** rescales the vector so that it has a **mean of 0** and a **variance of 1**.

This ensures that the inputs to the next layer are always in a consistent, well-behaved range, which helps speed up and stabilize training.

Let's normalize our residual output vector `z = [3.16, 0.61, 1.87]`.

**1. Calculate Mean ($\mu$):**
$\mu = (3.16 + 0.61 + 1.87) / 3 = 5.64 / 3 = 1.88$

**2. Calculate Variance ($\sigma^2$):**
$\sigma^2 = ((3.16-1.88)^2 + (0.61-1.88)^2 + (1.87-1.88)^2) / 3$
$\sigma^2 = (1.28^2 + (-1.27)^2 + (-0.01)^2) / 3 = (1.6384 + 1.6129 + 0.0001) / 3 = 1.08$

**3. Normalize the vector:**
The formula is `(z - μ) / sqrt(σ² + ε)`, where ε is a very small number to prevent division by zero. Let's use `sqrt(1.08) ≈ 1.04`.

* $z'_0 = (3.16 - 1.88) / 1.04 = 1.23$
* $z'_1 = (0.61 - 1.88) / 1.04 = -1.22$
* $z'_2 = (1.87 - 1.88) / 1.04 = -0.01$

Our normalized vector is `z_norm = [1.23, -1.22, -0.01]`.

### Scale and Shift
The normalization process is a bit restrictive. To give the network flexibility, we introduce two learnable parameter vectors: **scale** ($\gamma$) and **shift** ($\beta$). We multiply the normalized output by $\gamma$ and add $\beta$.

`Final Output = γ * z_norm + β`

If the network learns that $\gamma = \sqrt{\sigma^2}$ and $\beta = \mu$, it can effectively undo the normalization if it needs to. This gives the model the best of both worlds: stability by default, with the flexibility to learn the optimal distribution for its outputs.