---
title: Multilayer Perceptrons

---

# Multilayer Perceptrons

## Introduction
Inspired by the biological brain, neural networks are at the heart of modern machine learning. They decipher patterns in data, enabling tasks like image recognition and language understanding. At their core are *multilayer perceptrons (MLPs)*, feedforward networks that power a wide range of applications.

* **Perceptrons:** The simplest linear classifiers, taking multiple inputs and producing a single output. They categorize data into two groups based on a weighted sum of inputs.
* **MLPs:** Built by stacking multiple layers of interconnected perceptrons, MLPs introduce non-linearity and become powerful tools for complex pattern recognition.

### Key Applications
MLPs excel in various domains:

* **Image Recognition:** Identifying objects, faces, and scenes in pictures.
* **Natural Language Processing (NLP):** Analyzing sentiment in text, translating languages, and powering chatbots.
* **Speech Recognition:** Converting spoken words into text for applications like voice assistants.



## MLP Architecture: A Closer Look
* **Input Layer:** Receives raw data (e.g., pixel values of an image).
* **Hidden Layer(s):** Where the magic happens! Each neuron processes a weighted sum of inputs and applies an activation function. Multiple layers allow for increasingly abstract feature extraction.
* **Output Layer:** Produces the final results (e.g., probabilities of an image belonging to different classes).

![A-hypothetical-example-of-Multilayer-Perceptron-Network](https://hackmd.io/_uploads/Hkb9-UaA6.png)

### Neurons: The Computational Units of MLPs
Within the hidden and output layers of an MLP, neurons serve as the fundamental computational units. Each neuron receives a set of inputs, processes them, and generates an output. This process involves the following components:

* **Inputs:** Represented as a vector $\mathbf{x} = (x_1, x_2, \dots, x_n)^t \in \mathbb{R}^n$ , where each $x_i$ is a numerical value.
* **Weights:** Associated with each input is a weight $w_i \in \mathbb{R}$. These weights determine the importance or influence of each input on the neuron's output.
* **Bias:** A constant term $b \in \mathbb{R}$ that shifts the neuron's activation function, providing flexibility in modeling.

#### Neuron Calculation
The neuron first computes a weighted sum of its inputs and the bias:

$$
z = \mathbf{w}^t \mathbf{x} + b = \sum_{i=1}^{n} w_i x_i + b
$$

Then, it applies an activation function to this sum, producing the neuron's final output:

$$
\psi(\mathbf{x}) = \sigma(z)
$$

### Activation Functions: Introducing Non-Linearity
Activation functions are crucial for MLPs. They introduce non-linearity into the network, enabling it to learn and model complex patterns and relationships within data. Without activation functions, an MLP would be limited to representing only linear relationships, severely restricting its capabilities.

#### Common Activation Functions
* **Sigmoid (Logistic):** $\frac{1}{1+e^{-x}}$
    * Output range: $(0, 1)$
    * Primarily used for binary classification problems.
    * Can suffer from vanishing gradients during training.
* **Hyperbolic Tangent (Tanh):** $\frac{e^x - e^{-x}}{e^x + e^{-x}}$
    * Output range: $(-1, 1)$
    * Often preferred over sigmoid due to its centered output.
    * Can also experience vanishing gradients.
* **Rectified Linear Unit (ReLU):** $\max(0,x)$
    * Output range: $[0, \infty)$
    * Computationally efficient and mitigates the vanishing gradient problem.
    * Can lead to "dead" neurons that become inactive during training.
* **Softmax:** $\frac{e^{z_{j}}}{\sum_{k=1}^{K} e^{z_{k}}}$ (for a vector $\mathbf{z}$)
    * Output range: $(0, 1)$ for each element, and the sum of all elements is 1.
    * Commonly used in the output layer for multi-class classification tasks.
    * Provides a probability distribution over the possible classes.

#### Visualizing Activation Functions
        
```python=
import numpy as np
import matplotlib.pyplot as plt
 
x = np.linspace(-5,  5, 1000)
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.plot(x, 1/(1+np.exp(-x)),label = 'f(x)')
plt.title('sigmoid')
plt.axis([-5, 5, -1, 2])
plt.legend()
 
plt.subplot(2, 2, 2)
plt.plot(x, np.tanh(x))
plt.title('tanh')
plt.axis([-5, 5, -2, 2])
 
plt.subplot(2, 2, 3)
plt.plot(x[:500], np.zeros(500))
plt.plot(x[500:], x[500:],color = '#1f77b4')
plt.axis([-5, 5, -1, 2])
plt.title('ReLU')

plt.subplot(2, 2, 4)
plt.plot(x, np.exp(x)/(1+np.exp(x)))
plt.axis([-5, 5, -1, 2])
plt.title('softmax')
 
plt.show()
```
![act](https://hackmd.io/_uploads/SkW6k5pR6.png)

### Universal Approximation Theorem: Unlocking MLP's Power
The Universal Approximation Theorem states that a feedforward neural network with at least one hidden layer and a non-linear activation function can approximate any continuous function on a compact subset of the input space to arbitrary accuracy. This means MLPs are incredibly expressive models, capable of learning a vast array of complex functions.

#### Formal Statement
Let $K \subset \mathbb{R}^n$ be a compact set (e.g., a closed and bounded interval). Let $\sigma: \mathbb{R} \rightarrow \mathbb{R}$ be a continuous, bounded, and non-constant activation function. Then, for any continuous function $f:K \rightarrow \mathbb{R}$ and any $\epsilon>0$, there exists a two-layer neural network with activation function $\sigma$ that approximates $f$ to within $\epsilon$ error.

#### Key Points
* **Density:** The theorem guarantees the existence of a network that can approximate the function, not necessarily how to find it.
* **Non-Linearity:** The non-linear activation function is crucial for modeling complex relationships.
* **Practical Implications:** This theorem highlights the potential of MLPs and justifies their wide use in various machine learning tasks.



## Training MLPs: Backpropagation and Optimization
The goal of training an MLP is to find the optimal values for its weights and biases, enabling it to accurately predict outputs for given inputs. This process involves two key steps: forward propagation and backpropagation.

### Forward Propagation: Predicting Outputs
* **Input Layer:** Receives the input data and passes it to the first hidden layer.
* **Hidden Layer(s):** Each neuron in the hidden layers computes a weighted sum of its inputs, adds a bias term, and applies an activation function. This process is repeated for each hidden layer, with the output of one layer serving as the input for the next.
* **Output Layer:** The final layer processes the outputs of the last hidden layer and produces the network's predictions.

### Backpropagation: Adjusting Weights and Biases
1. **Loss Calculation:** The predicted outputs are compared to the true labels using a loss function, which measures the discrepancy between them. Common loss functions include *mean squared error (MSE)* for regression tasks and *cross-entropy loss* for classification tasks.
2. **Gradient Calculation:** Using the chain rule of calculus, the gradients of the loss function with respect to the weights and biases of each neuron are computed. These gradients indicate the direction and magnitude of change required to reduce the loss.
3. **Parameter Update:** An optimization algorithm, such as *stochastic gradient descent (SGD)*, uses the calculated gradients to update the weights and biases in the direction that minimizes the loss. This process is iteratively repeated until the model converges to a satisfactory performance level.

### Stochastic Gradient Descent (SGD): A Common Optimization Algorithm
SGD is a widely used optimization algorithm for training neural networks. It works by iteratively updating the model parameters based on the gradients calculated for a small subset (mini-batch) of the training data. This approach helps to avoid overfitting and allows for faster training on large datasets.

* **Initialization:** Model parameters are randomly initialized.
* **Iterative Optimization:**
    1. **Calculate Gradients:** For a mini-batch of data, compute the gradients of the loss function concerning the model parameters.
    2. **Update Parameters:** Adjust the parameters in the opposite direction of the gradients, scaled by the learning rate 

The parameter update rule can be summarized as:

$$
\mathbf{\Theta}_{t+1} 
= \mathbf{\Theta}_t - \eta \nabla L(\mathbf{\Theta}_t) 
= \mathbf{\Theta}_t -\frac{\eta }{N} \sum_{l=1}^{N}\nabla L_{l}(\mathbf{\Theta}_t) 
$$

where:
* $\mathbf{\Theta}_{t+1}$ represents the set of all model parameters at iteration $t$.
* $\eta$ is the learning rate.
* $L(\mathbf{\Theta}_t)$ is the gradient of the loss function with respect to the parameters.
* $N$ is the number of mini-batch size.

### Example: Single Hidden Layer MLP with MSE Loss
Let's consider a simple MLP with a single hidden layer, where the goal is to predict a continuous output value (regression task).

* Inputs: $x = (x_1, \dots, x_n)^t \in \mathbb{R}^n$
* Hidden Layer Outputs: $z_j = \sigma_1(\sum_{i=1}^{n}w_{j,i}x_i + b_j)$
* Output Layer Output: $\hat{y_l} = \sigma_2(\sum_{j=1}^{m}v_{l,j}z_j + c_l)$

We'll use the mean squared error (MSE) loss function:

$$
L(\mathbf{\Theta}) 
:= ||\hat{\mathbf{y}}-\mathbf{y}||^2 = \frac1p \sum^p_{l=1} (\hat{y}_l-y_l)^2
$$

Using backpropagation and the chain rule, we can calculate the gradients for the output and hidden layer weights:

* **Output Layer Weight $v_{l,i}$:** \begin{align*}
    \frac{\partial L(\mathbf{\Theta})}{\partial v_{l,i}}
    &= \sum_{l=1}^{p} \frac{\partial L(\mathbf{\Theta})}{\partial \hat{y_l}} \frac{\partial \hat{y_l}}{\partial v_{l,i}} \\
    &= \sum_{l=1}^{p} \frac{2}{p} (\hat{y_l} - y_l) \cdot \sigma_2^{'}(\sum_{j=1}^m{v_{l,j}z_j}+c) \cdot z_i
    \end{align*}
* **Hidden Layer Weight $w_{j,i}$:** \begin{align*}
    \frac{\partial L(\mathbf{\Theta})}{\partial w_{j,i}}
    &= \sum_{l=1}^{p} \frac{\partial L(\mathbf{\Theta})}{\partial \hat{y_l}} \frac{\partial \hat{y_l}}{\partial z_j} \frac{\partial z_j}{\partial w_{j,i}} \\
    &=  \sum_{l=1}^{p} \frac{2}{p}  (\hat{y_l}- y_l) \cdot \sigma_2^{'}(\sum_{j=1}^m{v_{l,j}z_j}+c))
    \cdot v_{l,j} \cdot \sigma_1^{'}(\sum_{i=1}^{n}{w_{j,i}x_i}+b) \cdot x_p
    \end{align*} 




## Combatting Overfitting in MLPs: Key Strategies
Overfitting occurs when a model learns the training data too well, capturing noise and fluctuations instead of the underlying patterns. This leads to poor generalization performance on unseen data. To mitigate this issue in MLPs, several effective strategies can be employed:

### Early Stopping
Early stopping involves monitoring the model's performance on a separate validation set during training. The training process is terminated when the model's performance on the validation set stops improving or starts to deteriorate. This prevents the model from over-optimizing on the training data and helps to retain its ability to generalize.

#### Implementation
1. Split the available data into training, validation, and test sets.
2. Train the model on the training set and evaluate its performance on the validation set after each epoch.
3. If the validation performance fails to improve for a specified number of consecutive epochs (patience), stop training.
4. Select the model with the best validation performance for final evaluation on the test set.

### Weight Decay (L2 Regularization)
Weight decay, also known as L2 regularization, adds a penalty term to the loss function that is proportional to the square of the weights. This penalty discourages the model from learning large weights, effectively preventing the model from becoming too complex and overfitting to the training data.

#### Loss Modification
The standard loss function, $L(\mathbf{\Theta})$, is modified by adding the regularization term:

$$
L^*(\mathbf{\Theta})=L(\mathbf{\Theta})+\frac{\lambda}{2}||\mathbf{\Theta}||^2
$$

where $\lambda$ is a hyperparameter that controls the strength of regularization. Larger values of $\lambda$ impose stronger penalties on large weights.

#### Parameter Update
The parameter update rule is modified accordingly:

$$
\mathbf{\Theta}_{t+1}:=(1-\eta\lambda)\mathbf{\Theta}_t - \eta \,\nabla L^{*}(\mathbf{\Theta_t})
$$

where $\eta$ is the learning rate.

### Dropout
Dropout is a regularization technique where, during each training iteration, a random fraction of neurons in each layer is temporarily "dropped out" or deactivated. This forces the network to learn more robust features and prevents co-adaptation of neurons, reducing overfitting.

#### Implementation
1. For each training iteration, sample a binary mask for each layer, where each element has a probability $p$ of being zero (dropped) and $1-p$ of being one (kept).
2. Multiply the layer's output by the corresponding mask.
3. Scale the remaining activations by $\frac{1}{1-p}$ to maintain the expected output magnitude.

![image](https://hackmd.io/_uploads/r1Sbegh1C.png)

#### Key Considerations
* Dropout is only applied during training. During evaluation, all neurons are used.
* The dropout rate $p$ is a hyperparameter that needs to be tuned. Typical values range from 0.2 to 0.5.



## Example: Recognizing Handwritten Digits with PyTorch
Let's solidify our understanding of MLPs by building a practical example in PyTorch. We'll create a model to classify handwritten digits from the MNIST dataset, a classic machine learning benchmark.

### Defining the MLP Architecture
The following `MLP` class defines our neural network:

```python=
import torch
import torch.nn as nn
import torch.optim as optim

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        self.layers.append(nn.ReLU())
        
        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            self.layers.append(nn.ReLU())
        
        # Output layer
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.layers.append(nn.LogSoftmax(dim=1))  # LogSoftmax for numerical stability

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
```

### Preparing the MNIST Dataset
```python=+
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split  # Import random_split

# Data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load the dataset
mnist_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)

# Split into training, validation, and testing sets
train_size = int(0.8 * len(mnist_dataset))  # 80% for training
val_size = len(mnist_dataset) - train_size  # Remaining 20% for validation

train_dataset, val_dataset = random_split(mnist_dataset, [train_size, val_size])
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)  # Validation loader
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### Training the Model
```python=+
# Define model, loss function, and optimizer
model = MLP(input_size=784, hidden_sizes=[128, 64], output_size=10)
criterion = nn.NLLLoss() 
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Lists to store metrics
train_losses = []
test_losses = []
train_accs = []
test_accs = []

# Training loop
for epoch in range(10):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device) 
        data = data.view(-1, 28*28)

        # Reset gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(data)

        # Calculate loss
        loss = criterion(output, target)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update training metrics
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

    train_loss /= len(train_loader)
    train_acc = 100. * correct / total

    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # Evaluate on test set
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(-1, 28*28)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / total

    test_losses.append(test_loss)
    test_accs.append(test_acc)
  
    # Display and store accuracy and loss every epoch
    print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
```

```
Epoch 1: Train Loss: 0.3614, Train Acc: 89.42%, Test Loss: 0.0027, Test Acc: 94.71%
Epoch 2: Train Loss: 0.1308, Train Acc: 95.92%, Test Loss: 0.0018, Test Acc: 96.41%
Epoch 3: Train Loss: 0.0868, Train Acc: 97.34%, Test Loss: 0.0016, Test Acc: 96.96%
Epoch 4: Train Loss: 0.0641, Train Acc: 97.97%, Test Loss: 0.0014, Test Acc: 97.29%
Epoch 5: Train Loss: 0.0496, Train Acc: 98.40%, Test Loss: 0.0013, Test Acc: 97.39%
Epoch 6: Train Loss: 0.0381, Train Acc: 98.73%, Test Loss: 0.0014, Test Acc: 97.37%
Epoch 7: Train Loss: 0.0302, Train Acc: 99.02%, Test Loss: 0.0013, Test Acc: 97.65%
Epoch 8: Train Loss: 0.0224, Train Acc: 99.29%, Test Loss: 0.0013, Test Acc: 97.74%
Epoch 9: Train Loss: 0.0165, Train Acc: 99.52%, Test Loss: 0.0013, Test Acc: 97.83%
Epoch 10: Train Loss: 0.0121, Train Acc: 99.69%, Test Loss: 0.0013, Test Acc: 97.78%
```

```python=+
# Plotting
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Training Accuracy')
plt.plot(test_accs, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.show()
```

![output](https://hackmd.io/_uploads/B16HBTt7R.png)


### Three Ways to Combat Overfitting
#### Early Stopping
```python=
best_val_loss = float('inf')  # Initialize best validation loss
epochs_no_improve = 0
patience = 3  # Number of epochs to wait before stopping

# ... Training loop same as before

    # Validation loop
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, target in val_loader:  # Use a separate validation dataloader
            data = data.view(-1, 28*28)
            output = model(data)
            val_loss += criterion(output, target).item()  # Sum up batch loss

    val_loss /= len(val_loader.dataset)  # Average validation loss

    # Check for improvement (Corrected indentation)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after epoch {epoch+1}")
            break

    print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
```

```
Epoch 1: Train Loss: 0.0015, Train Acc: 99.99%, Test Loss: 0.0013, Test Acc: 98.07%
Epoch 2: Train Loss: 0.0011, Train Acc: 100.00%, Test Loss: 0.0013, Test Acc: 98.03%
Epoch 3: Train Loss: 0.0009, Train Acc: 100.00%, Test Loss: 0.0013, Test Acc: 98.06%
Early stopping triggered after epoch 4
```

![output_early_stop](https://hackmd.io/_uploads/Hy0mvaYQC.png)

#### Weight Decay (L2 Regularization)
```python=
# Define model, loss function, and optimizer
model = MLP(input_size=784, hidden_sizes=[128, 64], output_size=10)
criterion = nn.NLLLoss()  # Use NLLLoss with LogSoftmax
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)  # Add weight_decay

# ... (rest of the training steps are the same)
```

```
Epoch 1: Train Loss: 0.3620, Train Acc: 89.34%, Test Loss: 0.0024, Test Acc: 95.22%
Epoch 2: Train Loss: 0.1280, Train Acc: 96.13%, Test Loss: 0.0018, Test Acc: 96.47%
Epoch 3: Train Loss: 0.0850, Train Acc: 97.32%, Test Loss: 0.0016, Test Acc: 96.85%
Epoch 4: Train Loss: 0.0636, Train Acc: 98.01%, Test Loss: 0.0016, Test Acc: 96.81%
Epoch 5: Train Loss: 0.0497, Train Acc: 98.48%, Test Loss: 0.0014, Test Acc: 97.38%
Epoch 6: Train Loss: 0.0370, Train Acc: 98.86%, Test Loss: 0.0012, Test Acc: 97.60%
Epoch 7: Train Loss: 0.0305, Train Acc: 99.00%, Test Loss: 0.0012, Test Acc: 97.79%
Epoch 8: Train Loss: 0.0226, Train Acc: 99.34%, Test Loss: 0.0012, Test Acc: 97.71%
Epoch 9: Train Loss: 0.0173, Train Acc: 99.51%, Test Loss: 0.0013, Test Acc: 97.52%
Epoch 10: Train Loss: 0.0139, Train Acc: 99.64%, Test Loss: 0.0013, Test Acc: 97.75%
```

![weight_decay_output](https://hackmd.io/_uploads/SyRvd6YmA.png)


#### Dropout
```python=
class Dropout_MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(Dropout_MLP, self).__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        self.layers.append(nn.ReLU())
        
        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            self.layers.append(nn.ReLU())
        
        # Output layer
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.layers.append(nn.LogSoftmax(dim=1))  # LogSoftmax for numerical stability

    def forward(self, x):
        for layer in self.layers[:-2]: # exclude the last two layers
            x = layer(x)
        x = nn.functional.dropout(x, p=0.5, training=self.training) # Dropout layer
        x = self.layers[-2](x) # Apply the second to last layer (linear)
        x = self.layers[-1](x) # Apply the last layer (logsoftmax)
        return x
```

```
Epoch 1: Train Loss: 0.5067, Train Acc: 84.64%, Test Loss: 0.0028, Test Acc: 94.53%
Epoch 2: Train Loss: 0.2144, Train Acc: 93.91%, Test Loss: 0.0020, Test Acc: 95.96%
Epoch 3: Train Loss: 0.1570, Train Acc: 95.47%, Test Loss: 0.0017, Test Acc: 96.80%
Epoch 4: Train Loss: 0.1257, Train Acc: 96.42%, Test Loss: 0.0017, Test Acc: 96.98%
Epoch 5: Train Loss: 0.1047, Train Acc: 97.04%, Test Loss: 0.0015, Test Acc: 97.36%
Epoch 6: Train Loss: 0.0912, Train Acc: 97.41%, Test Loss: 0.0014, Test Acc: 97.42%
Epoch 7: Train Loss: 0.0818, Train Acc: 97.62%, Test Loss: 0.0013, Test Acc: 97.71%
Epoch 8: Train Loss: 0.0707, Train Acc: 97.89%, Test Loss: 0.0015, Test Acc: 97.54%
Epoch 9: Train Loss: 0.0633, Train Acc: 98.13%, Test Loss: 0.0014, Test Acc: 97.67%
Epoch 10: Train Loss: 0.0581, Train Acc: 98.19%, Test Loss: 0.0014, Test Acc: 97.73%
```

![dropout_output](https://hackmd.io/_uploads/SJ-WK6tXR.png)




## Reference
* [The Universal Approximation Theorem](https://www.deep-mind.org/2023/03/26/the-universal-approximation-theorem)
* [Multilayer Perceptrons in Machine Learning: A Comprehensive Guide](https://www.datacamp.com/tutorial/multilayer-perceptrons-in-machine-learning )
* [PatternRecognitionAndDeepLearning-Lab](https://github.com/rocketeerli/PatternRecognitionAndDeepLearning-Lab/blob/master/DeepLearning-Lab/lab1/MLP.py)
* [機器學習- 神經網路(多層感知機 Multilayer perceptron, MLP)](https://chih-sheng-huang821.medium.com/機器學習-神經網路-多層感知機-multilayer-perceptron-mlp-含詳細推導-ee4f3d5d1b41)
