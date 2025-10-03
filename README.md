<div style="background-color:#1E293B; padding:20px; border-radius:10px; color:white;">
  <h1 align="center">🚀 Introduction to Building Large Language Models (LLMs) 🚀</h1>
</div>
<br>

Welcome to this repository! This guide is designed to demystify the process of building Large Language Models (LLMs) from the ground up. We'll break down each core component into simple, understandable steps, using a single running example to connect all the dots.

Whether you're a student, a developer, or just curious about AI, by the end of this guide, you'll have a solid conceptual understanding of how modern LLMs like GPT work.

## 🤔 What Are We Building?

### What is a Large Language Model (LLM)?
An LLM is a type of artificial intelligence that has been trained on a massive amount of text data. Its primary goal is to understand and generate human-like text. Think of it as a very sophisticated autocomplete system that can write essays, answer questions, translate languages, and even write code.

### What are Vectors?
In the world of AI, a **vector** is simply a list of numbers. We use vectors to represent complex things like words or sentences in a way that a computer can understand. For example, the word "king" might be represented by a vector `[0.9, 0.2, 0.1]` and "queen" by `[0.8, 0.3, 0.1]`. The numbers in the vector capture the word's meaning, context, and relationships with other words.

### What is a Transformer?
The **Transformer** is the revolutionary neural network architecture that powers most modern LLMs. Introduced in the 2017 paper "Attention Is All You Need," its key innovation was to process all words in a sentence simultaneously, rather than one by one. This parallel processing capability allowed it to be trained on much larger datasets, leading to the powerful models we see today. **The Transformer is the core of an LLM.**

### What is the Attention Mechanism?
The **Attention Mechanism** is the secret sauce of the Transformer. It allows the model to weigh the importance of different words in the input text when processing a specific word. For example, when translating the sentence "The cat sat on the mat," the model needs to know that "it" in a following sentence might refer to the "cat." Attention helps the model make these connections, even across long distances in the text. **Attention is the core of the Transformer.**

## 📚 Table of Contents

This repository is structured as a step-by-step guide. Each section builds upon the last.

* **[Section 1: Data Collection and Tokenization](./01_Data_and_Tokenization/README.md)**
    * *From raw text to numerical inputs the model can understand.*
* **[Section 2: Transformers and Self-Attention](./02_Transformers_and_Self_Attention/README.md)**
    * *A high-level overview of the engine that powers LLMs.*
* **[Section 3: Simplified Self-Attention](./03_Simplified_Self_Attention/README.md)**
    * *Calculating attention scores with dot products.*
* **[Section 4: Self-Attention with Trainable Weights](./04_Self_Attention_With_Weights/README.md)**
    * *Introducing learnable parameters to make attention smarter.*
* **[Section 5: Causal Attention](./05_Causal_Attention/README.md)**
    * *Ensuring the model can't cheat by looking at future words.*
* **[Section 6: Multi-Head Attention](./06_Multi_Head_Attention/README.md)**
    * *Allowing the model to focus on different things at once.*
* **[Section 7: Introducing Transformers](./07_Introducing_Transformers/README.md)**
    * *Assembling the full Transformer block architecture.*
* **[Section 8: Add & Normalize Layer](./08_Add_and_Normalize/README.md)**
    * *A crucial step for stabilizing and deepening the network.*
* **[Section 9: Feed-Forward Neural Network](./09_Feed_Forward_Network/README.md)**
    * *"Thinking" about the information gathered by attention.*
* **[Section 10: Add & Normalize Again](./10_Add_and_Normalize_Again/README.md)**
    * *Finishing the Transformer block with another stabilization step.*
* **[Section 11: Linear Layer and Projection](./11_Linear_Layer_and_Projection/README.md)**
    * *From rich context vectors to predicting the next word.*
* **[Section 12: Finding the Next Word](./12_Finding_the_Next_Word/README.md)**
    * *Strategies for generating coherent and creative text.*
* **[Section 13: Autoregressive Loop & KV Caching](./13_Autoregressive_Loop_and_KV_Cache/README.md)**
    * *Generating text word-by-word and optimizing the process.*
* **[Section 14: Pre Training Loop](./14_Pre_Training_Loop/README.md)**
    * *How to train the model to minimize loss.*
