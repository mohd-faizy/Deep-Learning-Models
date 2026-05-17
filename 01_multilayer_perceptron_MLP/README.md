# Multilayer Perceptron (MLP)

## Detailed Specification
A Multilayer Perceptron (MLP) is a foundational class of feedforward artificial neural network (ANN). An MLP consists of at least three layers of nodes: an input layer, a hidden layer, and an output layer. Except for the input nodes, each node is a neuron that uses a nonlinear activation function. MLPs utilize a supervised learning technique called backpropagation for training. Its multiple layers and non-linear activation distinguish MLP from a linear perceptron, enabling it to distinguish data that is not linearly separable.

## Technical Specification
- **Architecture**: Feedforward, fully connected.
- **Layers**: Input Layer, N Hidden Layers, Output Layer.
- **Activation Functions**: Typically ReLU, Sigmoid, Tanh for hidden layers; Softmax or Linear for output layer depending on the task.
- **Optimization**: Gradient descent-based methods (SGD, Adam, RMSprop).
- **Loss Functions**: Mean Squared Error (Regression), Cross-Entropy (Classification).

## The Mathematics
Given an input vector $x$, the output of a single hidden layer $h$ is computed as:
$$h = \sigma(W_1 x + b_1)$$
Where $W_1$ is the weight matrix, $b_1$ is the bias vector, and $\sigma$ is the activation function.
The final output $y$ for a two-layer MLP is:
$$y = \text{softmax}(W_2 h + b_2)$$

Training involves minimizing a loss function $L(y, \hat{y})$ using backpropagation, which computes the gradient of the loss function with respect to the weights using the chain rule:
$$\frac{\partial L}{\partial W_i} = \frac{\partial L}{\partial y} \frac{\partial y}{\partial h} \frac{\partial h}{\partial W_i}$$

## Pros
- **Universal Approximation**: Can theoretically approximate any continuous function given enough hidden units.
- **Simplicity**: Easy to understand, implement, and train.
- **Flexibility**: Can be adapted for regression, binary classification, and multi-class classification.

## Cons
- **Overfitting**: Highly prone to overfitting, especially on small datasets or with overly complex networks.
- **Lack of Spatial/Temporal Awareness**: Treats all input features independently; ignores spatial structures (like in images) and temporal sequences (like in text).
- **Computationally Expensive**: Fully connected layers require a massive number of parameters for high-dimensional inputs.

## Use Cases
- Tabular data analysis.
- Basic classification and regression tasks.
- Sensor data processing.
- Serving as the final classification layer in more complex architectures (like CNNs or Transformers).

## Limitations
- Very poor performance on raw image or audio data.
- Does not scale efficiently with high-dimensional input data due to $O(N \times M)$ parameter growth between layers.
