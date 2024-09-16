---
title: RNN
tags: [DL]

---

# Recurrent Neural Networks (RNNs)

## Introduction
Recurrent Neural Networks (RNNs) are a class of neural networks tailored for processing sequential data, where the order of elements is crucial.  By maintaining an internal memory, RNNs capture dependencies and patterns across time steps, making them ideal for tasks like natural language processing, time series analysis, and speech recognition.

### Key Concepts
* **Sequential Data:** Data with inherent order (e.g., words in a sentence, stock prices over time).
* **Hidden State:** RNN's internal memory, updating at each time step to retain relevant information.
* **Temporal Dependencies:** Relationships between data elements at different points within a sequence.

### Why RNNs?
* **Built for Sequences:** Designed to inherently handle ordered data.
* **Memory Capability:** Retains and utilizes information from earlier parts of a sequence.
* **Variable Length Handling:** Processes sequences of varying lengths.
* **Temporal Pattern Recognition:** Excels at identifying patterns across time steps.

### Applications
* **Natural Language Processing (NLP):** Machine translation, text generation, sentiment analysis.
* **Time Series Analysis:** Financial forecasting, weather prediction, anomaly detection.
* **Speech Recognition:** Transcription, voice commands.
* **Computer Vision:** Video analysis, action recognition.



## Architecture
* **Input $x_t$:** Data element at time step $t$.
* **Hidden State $h_t$:** Updated at each time step based on the current input and previous hidden state: $$h_t = f(W_{hh}h_{t-1} + W_{xh}x_t + b_h)$$
     * $f$: Non-linear activation function (e.g., tanh, ReLU)
    * $W_{hh}$, $W_{xh}$: Weight matrices
    * $b_h$: Bias term
* **Output $y_t$:** Prediction or output at time step $t$: $$y_t = g(W_{hy}h_t + b_y)$$
    * $g$: Output function (e.g., softmax for classification)
    * $W_{hy}$: Weight matrix
    * $b_y$: Bias term

![RNN](https://hackmd.io/_uploads/SJm5TaGlR.png)

### Challenges and Mitigation Strategies
RNNs, while powerful, come with some inherent challenges:

* **Vanishing/Exploding Gradients:** The recurrent nature of RNNs can lead to gradients that either diminish (vanish) or amplify (explode) exponentially during training. This hinders their ability to learn long-term dependencies effectively.
* **Training Complexity:** Compared to simpler feedforward networks, training RNNs can be more intricate and computationally demanding due to the need to backpropagate through time.

#### Mitigating Vanishing/Exploding Gradients
Several strategies have proven effective in addressing the vanishing/exploding gradient problem:

1. **Gradient Clipping:** This technique caps the norm (magnitude) of the gradient updates to a predefined threshold $\theta$. If the gradient's norm exceeds this threshold, it's scaled down. This prevents explosive gradients, stabilizing training.
$$
\mathbf{g}_{\text{clipped}} = 
\begin{cases} 
\mathbf{g} & \text{if } \|\mathbf{g}\| \leq \theta \\
\frac{\theta}{\|\mathbf{g}\|} \mathbf{g} & \text{if } \|\mathbf{g}\| > \theta 
\end{cases}
$$
2. **Batch Normalization:** By normalizing the activations within each mini-batch, batch normalization helps to maintain a stable distribution of values. This can prevent activations from saturating and vanishing gradients from occurring.
3. **Advanced Architectures (LSTM/GRU):** The Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) architectures are specifically designed to address the vanishing gradient problem. Their gating mechanisms provide more control over the flow of information, allowing them to capture dependencies across much longer sequences.



## Variants of RNNs
To overcome the limitations of standard RNNs, several specialized architectures have been developed:

### Long Short-Term Memory (LSTM)
* **Purpose:** Mitigates the vanishing/exploding gradient problem, enabling effective learning of long-term dependencies within sequences.
* **Mechanism:** Introduces a cell state alongside the hidden state, regulated by gating mechanisms to control information flow.

#### Architecture
![LSTM1](https://hackmd.io/_uploads/SkgCpaMeC.png)

* **Forget Gate:** Decides what information to discard from the cell state. $$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$
* **Input Gate:** Determines what new information to store in the cell state. \begin{split}i_t &= \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \\ \hat{C}_t &= \tanh(W_C \cdot [h_{t-1}, x_t] + b_C) \end{split}
* **Cell State Update:** Combines the old cell state with the new candidate values. $$C_t = f_t \odot C_{t-1} + i_t \odot \hat{C}_t$$
* **Output Gate:** Controls what information from the cell state is used to update the hidden state. \begin{split}o_t &= \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \\ h_t &= o_t \odot \tanh(C_t)\end{split} 

Here
* $h_t$: Hidden state at time $t$
* $C_t$: Cell state at time $t$
* $x_t$: Input at time $t$
* $\sigma$: Sigmoid function
* $\tanh$: Hyperbolic tangent function
* $W_f$, $W_i$, $W_C$, $W_o$: Weight matrices
* $b_f$, $b_i$, $b_C$, $b_o$: Bias terms

#### Code Implementation
```python=
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
```

### Gated Recurrent Unit (GRU)
* **Purpose:** A simplified alternative to LSTM, aiming for comparable performance with lower computational complexity.
* **Mechanism:** Combines the forget and input gates into a single update gate and merges the cell state and hidden state.

#### Architecture
![GRU](https://hackmd.io/_uploads/rJSa6aGeC.png)

* **Update Gate:** $$z_t = \sigma(W_z x_t + U_z h_{t-1} + b_z)$$
* **Reset Gate:** $$r_t = \sigma(W_r x_t + U_r h_{t-1} + b_r)$$
* **Candidate Hidden State:** $$\tilde{h}_t = \tanh(W_h x_t + U_h (r_t \odot h_{t-1}) + b_h)$$
* **Hidden State Update:** $$h_t = z_t \odot h_{t-1} + (1 - z_t) \odot \tilde{h}_t$$

Here
* $x_t$: Input vector at time $t$
* $h_t$: Hidden state vector at time $t$
* $z_t$: Update gate vector
* $r_t$: Reset gate vector
* $\tilde{h}_t$: Candidate hidden state vector
* $W_z$, $W_r$, $W_h$: Weight matrices for input-to-gate connections
* $U_z$, $U_r$, $U_h$: Weight matrices for hidden-to-gate connections
* $b_z$, $b_r$, $b_h$: Bias vectors

#### Code Implementation (PyTorch)
```python=
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out
```

### Deep RNN (DRNN)
* **Purpose:** Increases model capacity and captures complex patterns by stacking multiple RNN layers.
* **Mechanism:** Creates a hierarchical representation of the sequence, where each layer learns increasingly abstract features.

#### Architecture
![Depp_RNN.png.](https://hackmd.io/_uploads/SJQLDk7lA.png)

* **First Layer:** $$h^{(1)}_t = f^{(1)}(W^{(1)}_{hh}h^{(1)}_{t-1} + W^{(1)}_{xh}x_t + b^{(1)}_h)$$
* **Subsequent Layers ($l > 1$):** $$h^{(l)}_t = f^{(l)}(W^{(l)}_{hh}h^{(l)}_{t-1} + W^{(l)}_{xh}h^{(l-1)}_t + b^{(l)}_h)$$

Here
* $h^{(l)}_t$: Hidden state in layer $l$ at time $t$
* $f^{(l)}$: Activation function in layer $l$
* $W^{(l)}_{hh}$, $W^{(l)}_{xh}$: Weight matrices
* $b^{(l)}_h$ terms: Bias terms

#### Code Implementation
```python=
class DRNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out
```



## Example: Stock Price Prediction with RNNs
In this practical example, we'll illustrate how different RNN variants can be used for stock price prediction. We'll use historical stock price data for Netflix (NFLX) and apply the following steps:

1. **Data Preparation:** Gather and preprocess historical stock return data.
2. **Rolling Window Approach:** Divide the data into sliding windows for training and prediction.
3. **Model Training:** Train various RNN models (LSTM, GRU, DRNN) on the rolling windows.
4. **Evaluation and Comparison:** Assess and visualize the predictive performance of each model.

### Data Preparation
```python=
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

assets = 'NFLX'
period = "800d"
interval="1d"

class Dataset(Dataset):
    def __init__(self, assets, period, interval):
        self.assets = assets
        self.period = period
        self.interval = interval
        self.data = yf.download(self.assets, period=self.period, interval=self.interval)
        self.data['Return'] = self.data['Close'].pct_change().fillna(0)
        self.data['CR'] = (1+self.data['Return']).cumprod()
        
nflx = Dataset(assets, period, interval)

plt.figure(figsize=(10, 6))
plt.plot(nflx.data['CR'])
plt.title('NFLX cummulative return')
plt.show()
```
![image](https://hackmd.io/_uploads/B1SzTvANA.png)



### Rolling Window Approach
```python=+
def Dataloader_X(datas,T,batch_size,train_days):
    days = datas.shape[0]
    test_days = days - train_days - T

    N_T_data_train = np.array([datas[i:i+T] for i in range(train_days)])
    N_T_data_test = np.array([datas[i:i+T] for i in range(train_days,days-T)])
    A = [torch.tensor(N_T_data_train[i*batch_size:],dtype = torch.float).unsqueeze(2) if i == train_days//batch_size else torch.tensor(N_T_data_train[i*batch_size:(i+1)*batch_size],dtype = torch.float).unsqueeze(2) for i in range(train_days//batch_size+1)]
    print(A[0].size())
    print(A[-1].size())
    B = [torch.tensor(N_T_data_test[i*batch_size:],dtype = torch.float).unsqueeze(2) if i == test_days//batch_size else torch.tensor(N_T_data_test[i*batch_size:(i+1)*batch_size],dtype = torch.float).unsqueeze(2) for i in range(test_days//batch_size+1)]
    print(B[0].size())
    print(B[-1].size())
    return A,B

def Dataloader_Y(datas,T,batch_size,train_days):
    days = datas.shape[0]
    test_days = days - train_days - T
    datas = datas[T:]
    N_T_data_train = np.array([datas[i:i+1] for i in range(train_days)])
    N_T_data_test = np.array([datas[i:i+1] for i in range(train_days,days-T)])
    A = [torch.tensor(N_T_data_train[i*batch_size:],dtype = torch.float) if i == train_days//batch_size else torch.tensor(N_T_data_train[i*batch_size:(i+1)*batch_size],dtype = torch.float) for i in range(train_days//batch_size+1)]
    print(A[0].size())
    print(A[-1].size())
    B = [torch.tensor(N_T_data_test[i*batch_size:],dtype = torch.float) if i == test_days//batch_size else torch.tensor(N_T_data_test[i*batch_size:(i+1)*batch_size],dtype = torch.float) for i in range(test_days//batch_size+1)]
    print(B[0].size())
    print(B[-1].size())
    return A,B


days = data_.shape[0]
train_days = int(0.7*data_.shape[0])
window_size = 30
test_day = days - train_days - window_size
batch_size = 128
X_train, X_test = Dataloader_X(data_,window_size,batch_size,train_days)
Y_train, Y_test = Dataloader_Y(data_,window_size,batch_size,train_days)
```

### Model Training and Evaluation
```python=+
models = {'DRNN': model_DRNN, 'LSTM': model_LSTM, 'GRU': model_GRU,}  # Model dictionary
criterion = nn.MSELoss()  
epochs = 5000

for model_name, model in models.items():  # Iterate over variants
    criterion = nn.MSELoss()
    optimizer=torch.optim.Adam(model.parameters(), model.lr)
    print(f'Traning Model: {model_name}')
    losses_train = []
    losses_test = []
    for epoch in range(epochs):
        sum = 0
        for x,y in zip(X_train,Y_train):
            pred = model(x)
            loss_train = criterion(pred,y)
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            sum+=loss_train.item()
        losses_train.append(sum/len(X_train))
        sum = 0
        for x,y in zip(X_test,Y_test):
            pred = model(x)
            loss_test = criterion(pred,y)
            sum+=loss_test.item()
        losses_test.append(sum/len(X_test))
        if (epoch+1) % 500 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {losses_train[-1]:.8f}, Test Loss: {losses_test[-1]:.8f}')
```

```
Traning Model: DRNN
Epoch [500/5000], Train Loss: 0.00276692, Test Loss: 0.00308399
Epoch [1000/5000], Train Loss: 0.00156191, Test Loss: 0.00136096
Epoch [1500/5000], Train Loss: 0.00116587, Test Loss: 0.00098301
Epoch [2000/5000], Train Loss: 0.00093250, Test Loss: 0.00076772
Epoch [2500/5000], Train Loss: 0.00074309, Test Loss: 0.00061288
Epoch [3000/5000], Train Loss: 0.00060626, Test Loss: 0.00051424
Epoch [3500/5000], Train Loss: 0.00052874, Test Loss: 0.00045853
Epoch [4000/5000], Train Loss: 0.00048074, Test Loss: 0.00042106
Epoch [4500/5000], Train Loss: 0.00045862, Test Loss: 0.00040259
Epoch [5000/5000], Train Loss: 0.00045093, Test Loss: 0.00039587
```
![image](https://hackmd.io/_uploads/rJhlRv04R.png)
```
Traning Model: LSTM
Epoch [500/5000], Train Loss: 0.00499625, Test Loss: 0.00328889
Epoch [1000/5000], Train Loss: 0.00241228, Test Loss: 0.00238385
Epoch [1500/5000], Train Loss: 0.00176147, Test Loss: 0.00192160
Epoch [2000/5000], Train Loss: 0.00149930, Test Loss: 0.00146776
Epoch [2500/5000], Train Loss: 0.00120281, Test Loss: 0.00107446
Epoch [3000/5000], Train Loss: 0.00091105, Test Loss: 0.00078488
Epoch [3500/5000], Train Loss: 0.00069309, Test Loss: 0.00058762
Epoch [4000/5000], Train Loss: 0.00055483, Test Loss: 0.00052029
Epoch [4500/5000], Train Loss: 0.00048591, Test Loss: 0.00046742
Epoch [5000/5000], Train Loss: 0.00044448, Test Loss: 0.00043880
```
![image](https://hackmd.io/_uploads/BJ9E0wAV0.png)
```
Traning Model: GRU
Epoch [500/5000], Train Loss: 0.00131124, Test Loss: 0.00114364
Epoch [1000/5000], Train Loss: 0.00088335, Test Loss: 0.00074250
Epoch [1500/5000], Train Loss: 0.00066698, Test Loss: 0.00056189
Epoch [2000/5000], Train Loss: 0.00053948, Test Loss: 0.00046694
Epoch [2500/5000], Train Loss: 0.00046693, Test Loss: 0.00041148
Epoch [3000/5000], Train Loss: 0.00045271, Test Loss: 0.00040124
Epoch [3500/5000], Train Loss: 0.00044921, Test Loss: 0.00040107
Epoch [4000/5000], Train Loss: 0.00044543, Test Loss: 0.00040199
Epoch [4500/5000], Train Loss: 0.00044233, Test Loss: 0.00040468
Epoch [5000/5000], Train Loss: 0.00044024, Test Loss: 0.00040834
```
![image](https://hackmd.io/_uploads/S1rHAvAV0.png)

### Evaluation and Comparison
```python=+
pred_DRNN = []
pred_LSTM = []
pred_GRU = []
X = X_train
for batch  in X:
    for x in batch:
        pred = model_DRNN(x).detach().float()
        pred_DRNN.append(pred)
        pred = model_LSTM(x).detach().float()
        pred_LSTM.append(pred)
        pred = model_GRU(x).detach().float()
        pred_GRU.append(pred)
X= X_test
for batch in X:
    for x in batch:
        pred = model_DRNN(x).detach().float()
        pred_DRNN.append(pred)
        pred = model_LSTM(x).detach().float()
        pred_LSTM.append(pred)
        pred = model_GRU(x).detach().float()
        pred_GRU.append(pred)
print(pred_DRNN)
import matplotlib.pylab as plt
plt.figure(figsize=(18,6))
plt.plot(range(len(data_)),data_,label = 'original')
plt.plot(range(window_size,window_size+len(pred_DRNN[:train_days])),pred_DRNN[:train_days],label = 'DRNN_train')
plt.plot(range(window_size,window_size+len(pred_LSTM[:train_days])),pred_LSTM[:train_days],label = 'LSTM_train')
plt.plot(range(window_size,window_size+len(pred_GRU[:train_days])),pred_GRU[:train_days],label = 'GRU_train')
plt.plot(range(window_size+train_days-1,window_size+train_days-1+len(pred_DRNN[train_days-1:])),pred_DRNN[train_days-1:],label = 'DRNN_test')
plt.plot(range(window_size+train_days-1,window_size+train_days-1+len(pred_LSTM[train_days-1:])),pred_LSTM[train_days-1:],label = 'LSTM_test')
plt.plot(range(window_size+train_days-1,window_size+train_days-1+len(pred_GRU[train_days-1:])),pred_GRU[train_days-1:],label = 'GRU_test')
lables = nflx.data.index
step = 150
ticks = range(0,days,step)
plt.xticks(ticks,lables[ticks])
plt.ylabel('cummulated return')
plt.title('stock price vs predict price')
plt.legend()
plt.show()
```
![image](https://hackmd.io/_uploads/BJYORvRNC.png)
![image](https://hackmd.io/_uploads/SyMK0P0VA.png)

#### Use data from the first 30 days to predict the next 30 days 
```python=
pred_DRNN = []
pred_LSTM = []
pred_GRU = []
predict_days = 30
X = X_test[-1][-predict_days]
DRNN_x , LSTM_x,GRU_x = X,X,X

for i in range(predict_days):
    pred = model_DRNN(DRNN_x).detach().float()
    pred_DRNN.append(pred)
    DRNN_x = torch.stack((*DRNN_x[1:],pred),dim=0)

    pred = model_LSTM(LSTM_x).detach().float()
    pred_LSTM.append(pred)
    LSTM_x = torch.stack((*LSTM_x[1:],pred),dim=0)

    pred = model_GRU(GRU_x).detach().float()
    pred_GRU.append(pred)
    GRU_x = torch.stack((*GRU_x[1:],pred),dim=0)
```
![image](https://hackmd.io/_uploads/rJqLquCVA.png)


## Reference
* [CS 230 - Deep Learning (Stanford University)](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks)
* [Dive into Deep Learning - Recurrent Neural Networks](https://d2l.ai/chapter_recurrent-neural-networks/index.html)