---
title: Transformer
tags: [DL]

---

# Transformer

## Introduction
The Transformer architecture has revolutionized *natural language processing (NLP)* and is rapidly extending its influence into other fields. Its key innovation, the self-attention mechanism, allows models to directly understand relationships between elements within a sequence, regardless of their distance. This overcomes the limitations of traditional *recurrent neural networks (RNNs)* in handling long-range dependencies. The Transformer's success has led to breakthroughs in machine translation, text summarization, and many other NLP tasks.

### Motivation: Why Look Beyond RNNs?
*Recurrent neural networks (RNNs)* process sequences step-by-step, relying on a hidden state to carry information forward. While capable in theory, RNNs often struggle in practice due to:

* **Vanishing and Exploding Gradients:** These issues during backpropagation hinder learning long-term relationships.
* **Sequential Processing:** RNNs can't be fully parallelized, resulting in slower training times.
* **Limited Hidden State:** The fixed-size hidden state can lose information in long sequences.

### The Power of Attention
The attention mechanism provides a solution. Imagine accessing a database and querying specific information – this is how attention operates within the Transformer. It allows the model to selectively focus on the most relevant parts of the input for each prediction, creating "shortcuts" across the sequence. This mirrors how humans naturally focus on information during processing.



## What is Attention?
At its core, attention is a mechanism within neural networks that mimics human selective focus. In the Transformer:

* **Database:** Imagine a database of key-value pairs.
* **Query:** You have a query you want to find relevant information for.
* **Attention Process:** The mechanism compares your query to all keys, calculates "attention weights" (signifying relevance), and produces a weighted combination of the values.

### Formalizing Attention
Given a database $\mathcal{D} = \lbrace(\mathbf{k}_1, \mathbf{v}_1), \cdots, (\mathbf{k}_m, \mathbf{v}_m)\rbrace$ and a query vector ${\bf q}$, the attention function is:

$$
\text{Attention}(\mathbf{q}, \mathcal{D}) = \sum_{i=1}^m \alpha({\bf q}, \mathbf{k}_i) \mathbf{v}_i
$$

where $\alpha$ is a function outputting an attention weight.

#### Choices of Attention Weight
There are different ways to compute attention weights. Some common choices include:

$$
\begin{split}
\alpha(\mathbf{q}, \mathbf{k}) &= \exp\left(-\frac{1}{2} \|\mathbf{q} - \mathbf{k}\|^2\right) \qquad && \text{(Gaussian)} \\
\alpha(\mathbf{q}, \mathbf{k}) &= 
    \begin{cases} 
    1 & \text{if } \|\mathbf{q} - \mathbf{k}\| \leq 1 \\
    0 & \text{otherwise}
    \end{cases}
    && \text{(Boxcar)} \\
\alpha(\mathbf{q}, \mathbf{k}) &= \max(0, 1-\|\mathbf{q} - \mathbf{k}\|) && \text{(Epanechikov)}
\end{split}
$$

```python=
import matplotlib.pyplot as plt

# Define the functions
def gaussian(x):
    return np.exp(-x**2 / 2)
def boxcar(x):
    return np.abs(x) < 1.0
def epanechikov(x):
    return np.maximum(1 - np.abs(x), np.zeros_like(x))
# Generate x values
x_values = np.linspace(-3, 3, 400)
# Generate y values for each function
y_gaussian = gaussian(x_values)
y_boxcar = boxcar(x_values)
y_epanechikov = epanechikov(x_values)
# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x_values, y_gaussian, label='Gaussian')
plt.plot(x_values, y_boxcar, label='Boxcar')
plt.plot(x_values, y_epanechikov, label='Epanechnikov')
plt.title('Comparison of Attention Weight Functions')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
```

![output](https://hackmd.io/_uploads/S10nShuXA.png)

### Scaled Dot-Product Attention (The Transformer's Choice)
The Transformer uses a computationally efficient mechanism:

* **Scaled Dot Product:** $$α'(\mathbf{q}, \mathbf{k}_i) = \frac{\mathbf{q}^T \mathbf{k}_i}{\sqrt{d}}$$ (where $d$ is the dimension of the key vector) The scaling factor helps control magnitude and stabilize gradients.
* **Softmax Normalization:** $$α(\mathbf{q}, \mathbf{k}_i) = \text{softmax}\left(\frac{\mathbf{q}^T \mathbf{k}_i}{\sqrt{d}}\right) = \frac{\exp\left( \frac{\mathbf{q}^\mathsf{T} \mathbf{k}_i}{\sqrt{d}} \right)}{\sum_{j=1}^{m} \exp\left( \frac{\mathbf{q}^\mathsf{T} \mathbf{k}_j}{\sqrt{d}} \right)}$$
* **Weighted Values:** $$\text{Attention}(\mathbf{q}, \mathcal{D}) = \sum_{i=1}^m α(\mathbf{q}, \mathbf{k}_i) \mathbf{v}_i$$ This approach is efficient due to matrix operations and provides stable gradients.



## Self-Attention Mechanism
At the heart of the Transformer architecture lies the self-attention mechanism.  This mechanism allows the model to weigh the importance of different parts of the input sequence when making predictions for a specific element. Here's a breakdown of how it works:

### Queries, Keys, and Values
* **Input Embedding:** The process begins with an input sequence, such as a sentence, where each element (word) is represented as a vector embedding, denoted as $\{x_1, \dots, x_n\}$
* **Linear Transformations:** Three separate learnable weight matrices— $W_q$, $W_k$, and $W_v$—are used to project these input embeddings into three distinct vector spaces, producing:
    * **Queries Q:** $\mathbf{q}_i = W_q \mathbf{x}_i$. Queries represent what the model is looking for in relation to the current element.
    * **Keys K:** $\mathbf{k}_i = W_k \mathbf{x}_i$. Keys act as labels or identifiers for each element in the sequence.
    * **Values V:** $\mathbf{v}_i = W_v \mathbf{x}_i$. Values contain the actual information or content associated with each element.

### Calculating Attention
* **Scaled Dot-Product Attention:** The core of self-attention is the scaled dot-product attention mechanism. For each query $\mathbf{q}_i$, its attention with respect to all other elements is computed as: $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$ where:
    * $Q$ is the matrix of all queries.
    * $K$ is the matrix of all keys.
    * $V$ is the matrix of all values.
    * $d$ is the dimension of the query and key vectors.
    * The scaling factor $\frac{1}{\sqrt{d}}$ helps stabilize the gradients during training.
* **Output Representation:** The result of the attention calculation is a weighted sum of the values, where the weights are determined by the relevance of each key to the given query. This weighted sum forms a new representation of the element, incorporating information from other relevant parts of the sequence.



## Self-Attention Mechanism
At its heart, self-attention enables a Transformer model to understand how different parts of its input sequence relate to each other. Here's how it works:

<img src="https://hackmd.io/_uploads/r1TQLPF70.png" width="700" height="300">

### Queries, Keys, and Values
* **Embedding:** Start with an input sequence $\{\mathbf{x}_i\}$ (e.g., words in a sentence). Each input element (word) is represented as a vector embedding.
* **Transformation:** Three learned weight matrices $W_q, W_k, W_v$ are used to project the embeddings into three different vector spaces:
    * **Queries:** $\mathbf{q}_i = W_q \mathbf{x}_i$ represents what we're searching for within the input.
    * **Keys:** $\mathbf{k}_i = W_k \mathbf{x}_i$ act like labels that the queries will be compared against.
    * **Values:** $\mathbf{v}_i = W_v \mathbf{x}_i$ is the actual information or content we want to extract.


### Calculating Attention 
* For each query $\mathbf{q}_i$ with $\mathcal{D} = \lbrace(\mathbf{k}_1, \mathbf{v}_1), \cdots, (\mathbf{k}_n, \mathbf{v}_n)\rbrace$, $$\text{Attention}(\mathbf{q}_i, \mathcal{D}) = \sum_{j=1}^{n}\text{softmax}(\frac{\mathbf{q}_i \mathbf{k}_j^T}{\sqrt{d}})\mathbf{v}_j$$
* The self-attention process can be efficiently done using matrix operations $$\text{Attention}(Q,K,V) := \text{Softmax}(\frac{Q K^T}{\sqrt{d}})V = \big(\text{Attention}({\bf q}_i,\mathcal{D})\big),$$ making it computationally advantageous.

### Why It Matters
* **Long-Range Dependencies:** Capturing Long-Range Dependencies: Unlike recurrent neural networks (RNNs), which struggle to maintain information over long distances, self-attention can directly relate elements regardless of their separation in the sequence. This is crucial for language, where words far apart can have strong grammatical or semantic connections.
* **Parallelization:** The matrix operations involved in self-attention are highly parallelizable, making it much faster to compute than sequential models like RNNs, especially on modern hardware.



## Multi-Head Attention
In some cases, it's beneficial for a model to consider multiple perspectives on the same set of queries, keys, and values. This allows it to capture different types of relationships within a sequence—for example, both short-range and long-range dependencies. Multi-head attention achieves this by using multiple attention mechanisms operating in parallel, each focusing on a different representation subspace.

<img src="https://hackmd.io/_uploads/rkUI8DY7A.png"
width="700" height="300">


### Architecture
Instead of a single attention function with $d$-dimensional keys, values, and queries, multi-head attention projects these into $h$ different subspaces with dimensions $d_q$, $d_k$, and $d_v$, respectively. Attention is then computed in parallel on each of these projected versions, producing h output values, which are concatenated and linearly transformed into the final output.

Formally, multi-head attention can be expressed as:

$$
\text{MultiHead}(Q, K, V) = Concat(\text{head}_1, \dots, \text{head}_h)W^O
$$

where:

$$
\text{head}_i = \text{Attention}(QW^Q_i, KW^K_i, VW^V_i)
$$

* $Q$, $K$, and $V$ are the matrices of queries, keys, and values, respectively.
* $W^Q_i$, $W^K_i$, $W^V_i$ are the weight matrices that project the queries, keys, and values into the $i$-th subspace.
* $W^O$ is the weight matrix that transforms the concatenated outputs.

### Why Multi-Head Attention?
* **Diverse Perspectives:** Different heads can specialize in attending to different types of relationships within the sequence, leading to a more comprehensive understanding.
* **Enhanced Representational Power:** The combination of multiple heads provides a richer representation of the input sequence, improving the model's ability to capture complex patterns.
* **Parallelism:** The computations for each head can be performed in parallel, making multi-head attention computationally efficient.

### Difference Between Self-Attention and Multi-Head Attention
| Feature | Self-Attention | Multi-Head Attention |
| --- | --- | --- |
| Attention Mechanisms | Single | Multiple |
| Perspective | Single | Multiple, diverse |
| Representational Power | Limited | Enhanced |
| Computational Efficiency | High | High due to parallelism |

In essence:
* **Self-attention** gives a single, holistic view of the relationships within a sequence.
* **Multi-head attention** provides a more nuanced understanding by combining multiple perspectives.



## Cross Attention
Cross-attention is a powerful mechanism that enables models to establish connections between two distinct sources of information.  This is essential for tasks that require understanding and integrating information from multiple sequences or modalities.

Key applications of cross-attention include:
* **Sequence-to-Sequence Tasks:** For example, in machine translation, cross-attention helps the decoder (generating the target language) focus on relevant parts of the encoder's output (the source language).
* **Multi-modal Learning:** Cross-attention allows models to combine information from different sources, like images and text, to perform tasks like image captioning.
* **Adapting to Varying Input Lengths:** Cross-attention can dynamically adjust its focus based on the input, making it well-suited for sequences of varying lengths and complexities.

<img src="https://hackmd.io/_uploads/BJiKLPKmC.png"
width="700" height="300">

### Architecture
Cross-attention operates on two sets of input sequences:
* **Queries (Q):** Typically derived from the decoder's input or a separate sequence.
* **Keys (K) and Values (V):** Typically derived from the encoder's output or another sequence.

The process is as follows:
1. **Linear Transformations:** Weight matrices $W_q$, $W_k$, and $W_v$ are used to project the input vectors into the query, key, and value spaces, respectively.

\begin{align*}
{\bf q}_i &= W_q {\bf x}_{\text{Decoder},(i)}\\
{\bf k}_i &= W_k {\bf x}_{\text{Encoder},(i)}\\
{\bf v}_i &= W_v {\bf x}_{\text{Encoder},(i)}
\end{align*}

2. **Attention Calculation:** The attention mechanism (either scaled dot-product attention or multi-head attention) is then applied, using the decoder's queries to attend to the encoder's keys and values.
3. **Output:** The resulting weighted sum of the values is the output of the cross-attention mechanism. This can be used directly or further processed within the decoder.

### Key Advantages
* **Selective Focus:** Cross-attention allows the model to pinpoint the most relevant information in the key/value sequence while processing the query sequence.
* **Adaptability:** This selective focus is dynamic and can change based on the specific input, making cross-attention highly versatile.
* **Scalability:** Cross-attention is more efficient than attending to all elements in both sequences simultaneously, enabling the model to handle longer inputs effectively.



## Masked Attention
Masked attention is a variant of the attention mechanism specifically designed to prevent the model from attending to future tokens in a sequence. This is particularly important in decoder layers of Transformer models, where predictions are made autoregressively (one token at a time).

<img src="https://hackmd.io/_uploads/Hk4R8DtXA.png"
width="500" height="350">

### Architecture
1. **Standard Attention Calculation:** Queries (Q), keys (K), and values (V) are generated from the input sequence, and attention scores are computed using the scaled dot-product attention mechanism.
2. **Masking:** A mask matrix $M$ is applied to the attention scores before the softmax operation. The mask has the following structure: $$M = \begin{bmatrix}
    0 & -\infty & \cdots & -\infty \\
    0 & 0 & \cdots & -\infty \\
    \vdots & \vdots & \ddots & \vdots \\
    0 & 0 & \cdots & 0 
    \end{bmatrix}$$ This effectively sets the attention scores corresponding to future positions to negative infinity.
3. **Softmax and Output:**  After applying the mask, the softmax function is used to normalize the attention scores, and the resulting attention weights are used to compute a weighted sum of the values.  This ensures that the output at each position is influenced only by the current and past tokens.

### Mathematical Implementation
The masking process can be efficiently implemented by adding the mask matrix $M$ to the scaled dot-product attention scores before applying the softmax function:

$$
\text{Masked-Attention}(Q, K, V) = \text{Softmax}\left(\frac{(QK^T)}{\sqrt{d_k}} + M \right)V
$$

### Why Masking Matters
* **Preserves Information Flow:** Masking ensures that the model's predictions are based solely on information that is available at the current time step, preventing the model from "cheating" by peeking into the future.
* **Enables Autoregressive Generation:** In tasks like text generation, masking allows the model to generate text one token at a time, conditioned on the previously generated tokens.
* **Improves Model Performance:** In many cases, masking leads to better model performance by enforcing a more natural and meaningful information flow.



## Transformer Architecture
The Transformer architecture is a powerful and versatile framework for processing sequential data. It is composed of several key components, each designed to capture different aspects of information within a sequence. Let's break down its main building blocks:

<img src="https://hackmd.io/_uploads/Hy6CPDKmR.png"
width="500" height="600">

### Encoder
The encoder is responsible for processing the input sequence and creating a rich representation of its content. It consists of a stack of identical layers, where each layer performs two main operations:

1. **Multi-Head Self-Attention:** This mechanism allows the model to weigh the importance of different parts of the input sequence when processing a particular element. It helps the model capture long-range dependencies and understand the relationships between different parts of the input.
2. **Feedforward Neural Network:** This is a simple fully connected network applied independently to each position in the sequence. It helps the model learn complex non-linear relationships between the elements.

### Decoder
The decoder is designed to generate an output sequence, conditioned on the input sequence and its representation from the encoder. It also consists of a stack of identical layers, where each layer performs three main operations:

1. **Masked Multi-Head Self-Attention:** Similar to the self-attention in the encoder, but with a mask that prevents the model from attending to future positions. This ensures that the model can only use information available up to the current position when generating the next output element.
2. **Multi-Head Cross-Attention:** This mechanism allows the decoder to attend to the encoder's output, effectively focusing on the most relevant parts of the input sequence when generating the next output element.
3. **Feedforward Neural Network:** Identical to the feedforward network in the encoder, it helps the decoder learn complex relationships between the elements in the output sequence and the context from the encoder.

### Additional Components
* **Layer Normalization:** After each sublayer (self-attention and feedforward network), layer normalization is applied to stabilize training and help the model converge faster.
* **Residual Connections:** Each sublayer's output is added to its input (a technique called residual connection) before being passed to the next layer. This helps mitigate the vanishing gradient problem and allows for training deeper models.
* **Positional Encoding:** Since self-attention is permutation-invariant (doesn't care about the order of elements), positional encoding is added to the input embeddings to provide information about the position of each element in the sequence.

### Impact and Applications
The Transformer's impact has been truly transformative:

* **Natural Language Processing (NLP):** The Transformer has become the backbone of state-of-the-art models for a wide range of NLP tasks, including machine translation, text summarization, question answering, sentiment analysis, and language generation.
* **Beyond NLP:** The Transformer's versatility has led to its successful application in other domains, such as image recognition (Vision Transformers), music generation (Music Transformers), and even protein folding (AlphaFold).



## Example: The Nonlinear Autoregressive Exogenous (NARX) Model

### Problem Statement
The NARX model is designed to predict future values of a time series by considering both its own past values (autoregressive component) and the influence of external (exogenous) time series.

Formally:
* **Driving Series (Exogenous Inputs):** A set of $n$ time series, denoted as $X = (x^1, x^2, \ldots, x^n)^T$, each with a length of $T$ (the window size).
* **Target Series:** The time series we want to predict, denoted as $(y_1, y_2, \ldots, y_T)$.
* **Inputs at Time $t$:** The vector $x_t = (x_t^1, x_t^2, \ldots, x_t^n)^T$ (current values of the driving series) and past target values $(y_{t-1}, y_{t-2}, \dots)$.
* **Output:** The predicted value $y_{t+1}$.
* **Function:** The NARX model learns a nonlinear function $\hat{y}_{T+1} = F(y_1, \ldots, y_T, x_1, \ldots, x_T)$ to make predictions.

### Architecture
The NARX model typically consists of an encoder and a decoder, both incorporating attention mechanisms.

<img src="https://hackmd.io/_uploads/rk_M_PF7R.png"
width="700" height="250">

#### Encoder
1. **Input Attention:** This mechanism allows the model to weigh the importance of different driving series at each time step. It produces attention weights $\alpha_t^k$ that determine how much the $k$-th driving series contributes to the prediction at time $t$.
2. **LSTM Layer:** The encoder uses a Long Short-Term Memory (LSTM) layer to process the weighted input, producing a hidden state $h_t$ that captures the temporal dependencies in the data.

**Mathematical Formulation:** 
\begin{split}
e_t^k &= v_e^T \tanh(W_e [h_{t-1}; s_{t-1}] + U_e x^k) \\
\alpha_t^k &= \frac{\exp(e_t^k)}{\sum_{i=1}^n \exp(e_t^i)} \\
\tilde{x}_t &= (\alpha_t^1 x_t^1, \alpha_t^2 x_t^2, ..., \alpha_t^n x_t^n)^T \\
h_t &= \text{LSTM}_1(h_{t-1}, \tilde{x}_t)
\end{split} 

where 

- $h_{t-1}, s_{t-1}$  are the previous hidden state and cell state of the LSTM unit respectivly.
- $v_e \in \mathbb{R}^T$, $W_e \in \mathbb{R}^{T \times 2m}$ and $U_e \in \mathbb{R}^{T \times T}$ are parameters to learn.

#### Decoder
1. **Temporal Attention:** The decoder employs a temporal attention mechanism to focus on different time steps of the encoder's hidden states. This helps the model integrate information from the entire input sequence when making predictions.
2. **LSTM Layer:** The decoder's LSTM layer takes the context vector (a weighted sum of the encoder's hidden states) and the previous target value as input. It produces a hidden state that captures the relationships between the target series and the encoder's output.
3. **Output Layer:** A linear layer takes the decoder's hidden state and produces the final prediction $\hat{y}_{t+1}$.

**Mathematical Formulation:** 
\begin{split}
l_t^i &= v_d^T \tanh(W_d[d_{t-1}; s'_{t-1}] + U_d h_i) \\
\beta_t^i &= \frac{\exp(l_t^i)}{\sum_{j=1}^T \exp(l_t^j)} \\
c_t &= \sum_{i=1}^T \beta_t^i h_i \\
\tilde{y}_{t} &= \tilde{w}^T [y_{t}; c_{t}] + \tilde{b} \\
d_t &= \text{LSTM}_2(d_{t-1}, \tilde{y}_{t})
\end{split}

where 
- $d_{t-1}, s'_{t-1}$ are the previous hidden state and cell state of the LSTM unit respectivly.
- $v_d \in \mathbb{R}^m$, $W_d \in \mathbb{R}^{m \times 2p}$ and $U_d \in \mathbb{R}^{m \times m}$ are parameters to learn.
- $y_{t}, c_{t}$  are the decoder input and the computed context vector respectively
- $\tilde{w} \in \mathbb{R}^{m+1}$ and $\tilde{b} \in \mathbb{R}$ are parameters to learn.

#### Prediction
$$
\hat{y}_T = v_y^T (W_y [d_T; c_T] + b_w) + b_v
$$

where 
- $d_T, c_T$ are the decoder hidden state and the context vector of time step $T$.
- $W_y \in \mathbb{R}^{p \times (p+m)}$, $v_y \in \mathbb{R}^p$, $b_w \in \mathbb{R}^p$ and $b_v \in \mathbb{R}$ are parameters to learn.

### Code Implementation
We use cryptocurrency as the prediction target, aiming to predict BTC by utilizing the top 30 major cryptocurrencies. We download data from Bybit, obtaining 1,593 data points per minute, and split the dataset into training, validation, and test sets.

#### Imports and Setup
```python=
import torch             # Core deep learning library
import torch.nn as nn   # Neural network modules
import torch.optim as optim # Optimization algorithms
import pandas as pd     # Data manipulation
import numpy as np      # Numerical operations
from datetime import datetime 
import datetime as dt
import matplotlib.pyplot as plt # Plotting
from pybit.unified_trading import HTTP  # Bybit API interaction
import time
```

#### Model Definitions
```python=+
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, T):
        # input size: number of underlying factors
        # T: number of time steps (10)
        # hidden_size: dimension of the hidden state
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.T = T
        self.n = input_size
        self.lstm1= nn.LSTMCell(input_size=self.n, hidden_size=self.hidden_size).to(device)
        self.attnWe = nn.Linear(in_features = 2 * hidden_size, out_features = T, bias=False).to(device)
        self.attnUe = nn.Linear(in_features = T, out_features = T, bias=False).to(device)
        self.attnVe = nn.Linear(in_features = T, out_features = 1).to(device)

    def forward(self, input_data):
        # input_data: n*T
        # hidden, cell: initial states with dimention hidden_size
        encoder_output = torch.zeros((self.T, self.hidden_size)).to(device)
        lstm1_hidden_state = torch.zeros(self.hidden_size).to(device)
        lstm1_cell_state = torch.zeros(self.hidden_size).to(device)
        
        for t in range(self.T):
            
            # Attention calculation
            h_c_concat = torch.cat((lstm1_hidden_state, lstm1_cell_state))
            s1 = self.attnWe(h_c_concat.unsqueeze(0).repeat(self.n, 1))
            s2 = self.attnUe(input_data)
            s3 = torch.tanh(s1+s2)
            e_kt = self.attnVe(s3).squeeze()
            alpha_kt = F.softmax(e_kt,dim = 0)
            
            # Weighted input applied in LSTM
            x_tilde = alpha_kt * input_data.transpose(0,1)[t]
            lstm1_hidden_state, lstm1_cell_state = self.lstm1(x_tilde, (lstm1_hidden_state, lstm1_cell_state))
            
            # Encoder output
            encoder_output[t] = lstm1_hidden_state
            
        return encoder_output
```

```python=+
class Decoder(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size, T):
        super(Decoder, self).__init__()
        self.T = T
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.attnWd = nn.Linear(in_features = 2 * decoder_hidden_size, out_features = encoder_hidden_size, bias=False).to(device)
        self.attnUd = nn.Linear(in_features = encoder_hidden_size, out_features = encoder_hidden_size, bias=False).to(device)
        self.attnVd = nn.Linear(in_features = encoder_hidden_size, out_features = 1).to(device)
        self.lstm2 = nn.LSTMCell(input_size = 1, hidden_size = decoder_hidden_size).to(device)
        self.fc = nn.Linear(encoder_hidden_size + 1, 1).to(device)
        self.fc_final1 = nn.Linear(decoder_hidden_size + encoder_hidden_size, decoder_hidden_size).to(device)
        self.fc_final2 = nn.Linear(decoder_hidden_size, 1).to(device)

    def forward(self, encoder_output, target, mean, std):
        # input_data: n*T
        # hidden, cell: initial states with dimention decoder_hidden_size
        lstm2_hidden_state = torch.zeros(self.decoder_hidden_size).to(device)
        lstm2_cell_state = torch.zeros(self.decoder_hidden_size).to(device)
        
        for t in range(self.T):
            
            # Attention calculation
            h_c_concat = torch.cat((lstm2_hidden_state, lstm2_cell_state))
            s1 = self.attnWd(h_c_concat.unsqueeze(0).repeat(self.T,1))
            s2 = self.attnUd(encoder_output)
            s3 = torch.tanh(s1+s2)
            e_kt = self.attnVd(s3).squeeze()
            beta_kt = F.softmax(e_kt,dim = 0)
            
            # Context vector
            context = torch.sum(beta_kt.unsqueeze(1) * encoder_output, dim=0)
            
            # y_tilde calcaulation
            y_c_concat = torch.cat((target[t].unsqueeze(0), context))
            y_tilde = self.fc(y_c_concat)
            
            # Weighted input applied in LSTM
            lstm2_hidden_state, lstm2_cell_state = self.lstm2(y_tilde, (lstm2_hidden_state, lstm2_cell_state))
            
        # Decoder output
        d_c_concat = torch.cat((lstm2_hidden_state, context))
        decoder_output = self.fc_final2(self.fc_final1(d_c_concat))
        
        return std*decoder_output+mean
```

```python=+
class DA_RNN(nn.Module):
    def __init__(self, mean, std):
        super(DA_RNN, self).__init__()
        self.mean = mean
        self.std = std
        self.encoder = Encoder(30, 64, 10).to(device)
        self.decoder = Decoder(64, 64, 10).to(device)
        
    def forward(self, input_data, target):
        encoder_outputs = self.encoder(input_data)
        decoder_outputs = self.decoder(encoder_outputs, target, self.mean, self.std)
        return decoder_outputs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
```

#### Data Preparation Functions
```python=+
api_key = "your api key"
api_secret = "your api secret"
session = HTTP(api_key=api_key, api_secret=api_secret, testnet=False)
```

#### Data Download and Preprocessing
```python=+
def get_last_timestamp(df):
    
    return int(df.timestamp[-1:].values[0])

def format_data(response):
    
    data = pd.DataFrame(response, columns =['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
    f = lambda x: dt.datetime.utcfromtimestamp(int(x)/1000)
    data.index = data.timestamp.apply(f)
    return data[::-1].apply(pd.to_numeric)

def download_as_pd(symbol, interval, start, end):
    df = pd.DataFrame()
    while True:
        response = session.get_kline(category='linear', 
                                     symbol=symbol, 
                                     start=start,
                                     interval=interval,
                                     timeout=30).get('result').get('list')
        
        latest = format_data(response)
        start = get_last_timestamp(latest)
        time.sleep(0.1)
        df = pd.concat([df, latest])
        if start > end: break
    df.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)
    return df

def normalization(data):
    data = np.array(data)
    mean = np.mean(data)
    std = np.std(data)
    batch_normalized_data = (data-mean)/std
    return batch_normalized_data
```

```python=+
cryptos_list = ["ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "TONUSDT", "ADAUSDT", "SHIB1000USDT", "AVAXUSDT", "TRXUSDT", "DOTUSDT", "BCHUSDT", "LINKUSDT", "NEARUSDT", "MATICUSDT", "LTCUSDT", "ICPUSDT", "HNTUSDT", "UNIUSDT", "ETCUSDT", "APTUSDT", "HBARUSDT", "CROUSDT", "STXUSDT", "MNTUSDT", "XLMUSDT", "FILUSDT", "ATOMUSDT", "1000PEPEUSDT", "BANDUSDT"]
datas = [] #size: n*T
standardized_datas = []
for crypto in cryptos_list:
    seq = download_as_pd(crypto, 1, int(dt.datetime(2024, 5, 1).timestamp()* 1000), int(dt.datetime(2024, 5, 11).timestamp()* 1000))
    print(crypto)
    print(seq['close'])
    datas.append(seq['close'])
    standardized_datas.append(normalization(seq['close']))
```

![BTC](https://hackmd.io/_uploads/Hks8qPYQ0.png)

#### Data Loader Functions
```python=+
def dataloader_X(X, batch_size, window_size):
    # X: numpy(input_size, T)
    data = torch.tensor([X.T[i:i+window_size].T for i in range(X.shape[1]-window_size+1)], dtype=torch.float, requires_grad=True)
    datas = []
    for i in range(X.shape[1]//batch_size+1):
        if i == X.shape[1]//batch_size:
            datas.append(data[i*batch_size:])
        else:
            datas.append(data[i*batch_size:(i+1)*batch_size])
    if window_size == 1:
        datas = [data.view((data.size()[0],1)) for data in datas]
    return datas

def dataloader_Y(Y, batch_size, window_size):
    # Y: numpy(T)
    data = torch.tensor([Y[i:i+window_size] for i in range(Y.shape[0]-window_size+1)], dtype=torch.float, requires_grad=True)
    datas = []
    for i in range(Y.shape[0]//batch_size+1):
        if i == Y.shape[0]//batch_size:
            datas.append(data[i*batch_size:])
        else:
            datas.append(data[i*batch_size:(i+1)*batch_size])
    if window_size == 1:
        datas = [data.view((data.size()[0],1)) for data in datas]
    return datas

def dataloader_Z(Z, batch_size, window_size):
    # Z: numpy(T)
    data = torch.tensor([Z[i:i+window_size] for i in range(Z.shape[0]-window_size+1)], dtype=torch.float, requires_grad=True)
    datas = []
    for i in range((Z.shape[0] // batch_size) + 1):
        if i == Z.shape[0] // batch_size:
            datas.append(data[i*batch_size:])
        else:
            datas.append(data[i*batch_size:(i+1)*batch_size])
    datas = [data.view(data.size()[0],1) for data in datas]
    return datas
```

#### Data Loader Creation
```python=+
# Data loaders
window_size = 10
batch_size = 128
input_size = np.array(datas).shape[0]
T = np.array(datas).shape[1]

# Datas size: n*T
X = np.array(standardized_datas)
Y = np.array(standardized_target)
Z = np.array(target[window_size:])

train_ratio = 0.7
validation_ratio = 0.15
test_ratio = 0.15

# train_day+validation_day+test_day = T-1-window_size
train_day = int(T*train_ratio)
validation_day = int(T*validation_ratio)
test_day = int(T*test_ratio)

X_train_data = dataloader_X(X.T[:train_day].T,batch_size,window_size)
X_validation_data = dataloader_X(X.T[train_day-window_size+1:train_day+validation_day].T,batch_size,window_size)
X_test_data = dataloader_X(X.T[train_day+validation_day-window_size+1:train_day+validation_day+test_day].T,batch_size,window_size)

Y_train_data = dataloader_Y(Y[:train_day],batch_size,window_size)
Y_validation_data = dataloader_Y(Y[train_day-window_size+1:train_day+validation_day],batch_size,window_size)
Y_test_data = dataloader_Y(Y[train_day+validation_day-window_size+1:train_day+validation_day+test_day],batch_size,window_size)

labels_train_data = dataloader_Z(Z[:train_day-window_size+1],batch_size,1)
labels_validation_data = dataloader_Z(Z[train_day-window_size+1:train_day+validation_day-window_size+1], batch_size,1)
labels_test_data = dataloader_Z(Z[train_day+validation_day-window_size+1:train_day+validation_day+test_day-window_size+1],batch_size,1)

print("train_day:",train_day)
print("validation_day:",validation_day)
print("test_day:",test_day)
print("====================")
print("Shape of Each Input")
print("X.shape:", X.shape)
print("Y.shape:", Y.shape)
print("Z.shape:", Z.shape)
print("====================")
print("Training Data")
print(len(X_train_data))
print(X_train_data[-1].size())
print(len(Y_train_data))
print(Y_train_data[-1].size())
print(len(labels_train_data))
print(labels_train_data[-1].size())
print("====================")
print("Validation Data")
print(len(X_validation_data))
print(X_validation_data[-1].size())
print(len(Y_validation_data))
print(Y_validation_data[-1].size())
print(len(labels_validation_data))
print(labels_validation_data[-1].size())
print("====================")
print("Test Data")
print(len(X_test_data))
print(X_test_data[-1].size())
print(len(Y_test_data))
print(Y_test_data[-1].size())
print(len(labels_test_data))
print(labels_test_data[-1].size())
```

```
train_day: 10169
validation_day: 2179
test_day: 2179

Shape of Each Input
X.shape: (30, 14528)
Y.shape: (14528,)
Z.shape: (14518,)

Training Data
80
torch.Size([48, 30, 10])
80
torch.Size([48, 10])
80
torch.Size([48, 1])

Validation Data
18
torch.Size([3, 30, 10])
18
torch.Size([3, 10])
18
torch.Size([3, 1])

Test Data
18
torch.Size([3, 30, 10])
18
torch.Size([3, 10])
18
torch.Size([3, 1])
```

#### Model Training
``` python=+
def trian(model, X_train_data, Y_train_data, labels_train_data, X_validation_data, Y_validation_data, labels_validation_data, criterion, optimizer, epoch_scheduler, epoches):
    
    hist_train_losses = []
    hist_validation_losses = []
    test = []
    
    for epoch in range(epoches):
        
        # Training training
        model.train()
        train_losses = []
        
        for batch_x, batch_y, label in zip(X_train_data, Y_train_data, labels_train_data):
            output = torch.zeros(batch_x.size(0)).to(device)
            for i in range(batch_x.size(0)):
                output[i] = model(batch_x[i].to(device), batch_y[i].to(device))
            train_loss = criterion(output.unsqueeze(1).to(device), label.to(device))
            train_losses.append(train_loss.cpu().detach().numpy())
            test.append(train_loss.cpu().detach().numpy())
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        hist_train_losses.append(np.mean(train_losses))
        
        # Validation training
        model.eval()
        Validation_outputs = []
        for batch_x, batch_y in zip(X_validation_data, Y_validation_data):
            for data_x, data_y in zip(batch_x, batch_y):
                Validation_outputs.append(model(data_x.to(device), data_y.to(device)).cpu().detach().float())
        validation_loss = criterion(torch.tensor(Validation_outputs, dtype=torch.float).to(device), torch.tensor(Z[train_day-window_size+1:train_day+validation_day-window_size+1]).unsqueeze(1).to(device))
        hist_validation_losses.append(validation_loss.cpu().detach().numpy())
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epoches}], Train Loss: {hist_train_losses[epoch]:.8f}, Validation Loss: {hist_validation_losses[epoch]:.8f}')
            
        epoch_scheduler.step()
    
    plt.figure(figsize=(10, 6))
    plt.plot(hist_train_losses,label='Train Loss')
    plt.plot(hist_validation_losses,label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoches')
    plt.ylabel('Loss') 
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(10, 6)) 
    plt.plot(Validation_outputs,label='Predict')
    plt.plot(Z[train_day-window_size+1:train_day+validation_day-window_size+1],label='Target')
    plt.title('Prediction')
    plt.xlabel('Time')
    plt.ylabel('Closed price') 
    
    lables = df_BTC.index[train_day-window_size+1+1:train_day+validation_day-window_size+1+1]
    step = 800
    ticks = range(0,validation_day,step)
    plt.xticks(ticks,lables[ticks])
    plt.legend()
    plt.show()
```

```python=+
model = DA_RNN(target_mean,target_std).to(device)
criterion = nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), 0.001)
epoch_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
trian(model, X_train_data, Y_train_data, labels_train_data, X_validation_data, Y_validation_data, labels_validation_data, criterion, optimizer, epoch_scheduler, epoches=300)
```

```
Epoch [10/130], Train Loss: 398799.03125000, Validation Loss: 2184248.56911372
Epoch [20/130], Train Loss: 65896.32812500, Validation Loss: 920649.45557177
Epoch [30/130], Train Loss: 14307.95019531, Validation Loss: 758672.40965541
Epoch [40/130], Train Loss: 6400.70214844, Validation Loss: 670651.50138337
Epoch [50/130], Train Loss: 8116.63183594, Validation Loss: 692032.19140182
Epoch [60/130], Train Loss: 11276.28906250, Validation Loss: 791499.70613630
Epoch [70/130], Train Loss: 6853.82275391, Validation Loss: 701995.32412715
Epoch [80/130], Train Loss: 4545.63378906, Validation Loss: 657723.02864535
Epoch [90/130], Train Loss: 4069.93823242, Validation Loss: 662634.98301253
Epoch [100/130], Train Loss: 4568.31494141, Validation Loss: 658677.70744753
Epoch [110/130], Train Loss: 3809.64135742, Validation Loss: 668149.03247149
Epoch [120/130], Train Loss: 3417.95190430, Validation Loss: 656192.76682351
Epoch [130/130], Train Loss: 3358.88354492, Validation Loss: 653772.66759277
```

![Loss](https://hackmd.io/_uploads/SyYgeM9QR.png)

![Prediction](https://hackmd.io/_uploads/HJEXgf97A.png)

#### Results and Visualization
```python=+
model.eval()
pred_list = []
for i in range(train_day):
    output = model(torch.tensor(X[:,i:i+window_size], dtype=torch.float),torch.tensor(Y[i:i+window_size], dtype=torch.float))
    pred_list.append(output.detach().numpy())
for i in range(validation_day):
    output = model(torch.tensor(X[:,train_day-window_size +1+i:train_day-window_size+1+i+window_size], dtype=torch.float),torch.tensor(Y[train_day-window_size+1+i:train_day-window_size+1+i+window_size], dtype=torch.float))
    pred_list.append(output.detach().numpy())
for i in range(test_day):
    output = model(torch.tensor(X[:,train_day+validation_day-window_size+1+i:train_day+validation_day-window_size+1+i+window_size], dtype=torch.float),torch.tensor(Y[train_day+validation_day-window_size+1+i:train_day+validation_day-window_size+1+i+window_size], dtype=torch.float))
    pred_list.append(output.detach().numpy())

plt.figure(figsize=(18,6))
plt.plot(range(train_day+validation_day+test_day),target_std*Y[:train_day+validation_day+test_day]+target_mean,label='Target')
plt.plot(range(window_size,len(pred_list[:train_day])+window_size),pred_list[:train_day],label='Train_predict')
plt.plot(range(window_size+train_day,len(pred_list[train_day-1:train_day+validation_day])+window_size+train_day),pred_list[train_day-1:train_day+validation_day],label='Validation_predict')
plt.plot(range(window_size+train_day+validation_day,len(pred_list[train_day+validation_day-1:train_day+validation_day+test_day])+window_size+train_day+validation_day),pred_list[train_day+validation_day-1:train_day+validation_day+test_day],label='Test_predict')

lables = df_BTC.index
step = 2000
ticks = range(0,train_day+validation_day+test_day,step)

plt.xticks(ticks,lables[ticks])
plt.title('Prediction')
plt.xlabel('Time')
plt.ylabel('Closed price')
plt.legend()
plt.show()
```

![Model_Prediction](https://hackmd.io/_uploads/H1NHlz57C.png)




## Reference
- [Attention Is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)
- [Transformer Dissection: An Unified Understanding for Transformer’s Attention via the Lens of Kernel](https://aclanthology.org/D19-1443/)
- [A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction](https://www.ijcai.org/proceedings/2017/366)

