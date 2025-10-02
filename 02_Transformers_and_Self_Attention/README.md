<div style="background-color:#4F46E5; padding:20px; border-radius:10px; color:white;">
  <h1 align="center">Section 2: Transformers and Self-Attention</h1>
</div>
<br>

Now that we have our input vectors (word meaning + position), we feed them into the heart of the LLM: the **Transformer**.

## 🏗️ The Transformer Architecture

A Transformer is not a single, monolithic block. It's a stack of identical layers, called **Transformer Blocks**. Each block processes the input vectors and passes its output to the next block.

> **Analogy:** Think of it like an assembly line. Each Transformer Block is a station that refines the product. Our initial vectors are the raw materials, and each block adds more context and understanding until we get a final, polished output.

**The Transformer is the core of an LLM.** It's the architecture that enables models to be trained at a massive scale.

## ✨ The Magic Ingredient: Self-Attention

Inside each Transformer Block, the most important component is the **Self-Attention** mechanism.

> **Attention is the core of the Transformer.**

Self-attention allows the model to weigh the importance of different words in the input sequence when processing a specific word. It helps the model build a "context-rich" understanding of each word.

For our sentence, **"I am studying artificial intelligence engineering"**, when the model processes the word "intelligence," self-attention might help it determine that "artificial" is highly relevant, while "I" and "am" are less so. It creates connections between words.



---

### The Four Flavors of Attention

We will build our understanding of attention step-by-step. There are several variations, each adding a layer of complexity and power.

1.  <div style="background-color:#F3F4F6; padding:10px; border-radius:5px; margin-bottom:10px;">
    <strong>Simplified Self-Attention (No Weights)</strong><br>
    The most basic form. We use the input vectors directly to see how much each word relates to every other word. This helps us understand the fundamental math.
    </div>

2.  <div style="background-color:#F3F4F6; padding:10px; border-radius:5px; margin-bottom:10px;">
    <strong>Self-Attention with Trainable Weights</strong><br>
    This is the real deal. We introduce learnable weight matrices (Wq, Wk, Wv) that allow the model to *learn* what to pay attention to during training. It makes the attention mechanism flexible and powerful.
    </div>

3.  <div style="background-color:#F3F4F6; padding:10px; border-radius:5px; margin-bottom:10px;">
    <strong>Causal Attention (Masked Self-Attention)</strong><br>
    Essential for language generation. This version prevents a word from "seeing" future words. When predicting the word after "studying," the model should only have access to "I," "am," and "studying," not the words that come after. It enforces a "causal" flow of information.
    </div>

4.  <div style="background-color:#F3F4F6; padding:10px; border-radius:5px; margin-bottom:10px;">
    <strong>Multi-Head Causal Attention</strong><br>
    The final form used in models like GPT. Instead of having one attention mechanism, we have multiple "heads" running in parallel. Each head can learn to focus on different types of relationships (e.g., one head for grammatical relationships, another for semantic ones). This gives the model a much richer understanding of the text.
    </div>

In the next sections, we will dive deep into each of these, starting with the simplest form.