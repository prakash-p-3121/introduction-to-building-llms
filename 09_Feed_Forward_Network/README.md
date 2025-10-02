<div style="background-color:#9D174D; padding:20px; border-radius:10px; color:white;">
  <h1 align="center">Section 9: Feed-Forward Neural Network</h1>
</div>
<br>

After the first "Add & Norm" step, our vectors have gathered contextual information from other tokens via the attention mechanism. Now, they need to be processed or "thought about" individually. This is the job of the **Position-wise Feed-Forward Network (FFN)**.

"Position-wise" simply means that the exact same network is applied to each token's vector independently. The vector for "studying" goes through the FFN, and the vector for "intelligence" goes through the exact same FFN, but they don't interact with each other in this step.

## 🧠 The FFN Architecture

The FFN in a Transformer is a simple two-layer fully connected neural network. It consists of:

1.  **An Expansion Layer:** The first linear layer expands the dimension of the input vector. A standard practice is to expand it by a factor of 4.
2.  **A Non-Linear Activation Function:** Typically **ReLU** (Rectified Linear Unit).
3.  **A Contraction Layer:** The second linear layer projects the expanded vector back down to the original input dimension.

The formula looks like this:
$$ FFN(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2 $$

### Expansion and Contraction

Let's say our input vector from the "Add & Norm" layer has a dimension of 3 (in reality, this is the `d_model` like 768).

* **Expansion:** The first layer's weight matrix $W_1$ would transform the dimension from 3 to, for example, $3 \times 4 = 12$.
* **Contraction:** The second layer's weight matrix $W_2$ would transform the dimension back down from 12 to 3.

**Why do this?** This expansion-contraction structure allows the model to learn more complex relationships and features from the data within a higher-dimensional space before compressing that knowledge back down. It gives the model more capacity to "think."

**In real models:**
For GPT-3, the model dimension (`d_model`) is 12288. The FFN expands this to `4 * 12288 = 49152` before contracting it back to 12288.

---

## ⚡ The ReLU Activation Function

Between the two linear layers, we apply a non-linear activation function. The most common one is **ReLU**.

`ReLU(x) = max(0, x)`

It's a very simple function:
* If the input `x` is positive, the output is `x`.
* If the input `x` is negative, the output is `0`.

Let's say after the first linear layer, one of our expanded vectors is `[2.5, -1.8, 0.0, 5.1]`.

Applying ReLU gives: `[max(0, 2.5), max(0, -1.8), max(0, 0.0), max(0, 5.1)] = [2.5, 0.0, 0.0, 5.1]`

**Why is this essential?** Without a non-linear activation function, stacking linear layers is pointless. A sequence of linear operations is mathematically equivalent to a single linear operation. The non-linearity is what allows the network to learn complex patterns that are not simply linear combinations of the inputs. It's a fundamental building block of deep learning.

The output of this FFN is a newly processed vector for each token. This vector is then passed to the final "Add & Norm" layer of the Transformer block.