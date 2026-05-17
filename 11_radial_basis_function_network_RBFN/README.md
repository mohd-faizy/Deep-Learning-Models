# Radial Basis Function Network (RBFN)

## Detailed Specification
A Radial Basis Function Network (RBFN) is an artificial neural network that uses radial basis functions as activation functions. It typically has three layers: an input layer, a hidden layer with a non-linear RBF activation function, and a linear output layer. RBFNs are renowned for their ability to universally approximate continuous functions and are often trained much faster than standard Multi-Layer Perceptrons (MLPs).

## Technical Specification
- **Architecture**: Feedforward, three layers (Input, Hidden RBF layer, Linear Output).
- **Activation Function**: Radial Basis Function, typically the Gaussian function. The response of a hidden neuron depends on the distance between the input vector and a center vector associated with the neuron.
- **Training (Two-Phase)**:
  1. **Unsupervised**: Determine the centers and widths of the RBFs (e.g., using K-Means clustering).
  2. **Supervised**: Train the weights of the linear output layer (often using singular value decomposition or simple pseudo-inverse, which is extremely fast).

## The Mathematics
For an input vector $x$, the output of the $j$-th hidden neuron is the radial basis function:
$$\phi_j(x) = \exp\left(-\frac{||x - c_j||^2}{2\sigma_j^2}\right)$$
Where $c_j$ is the center vector for neuron $j$, and $\sigma_j$ is its spread or width.
The final output of the network $y_i$ is a linear combination of these RBF outputs:
$$y_i = \sum_{j=1}^M w_{ij} \phi_j(x) + b_i$$
Where $w_{ij}$ are the weights from the hidden to output layer, and $b_i$ is the bias.

## Pros
- **Fast Training**: The two-stage training process (clustering followed by solving a linear system) is significantly faster than backpropagation used in MLPs.
- **Universal Approximation**: Guaranteed to approximate any continuous function given enough hidden neurons.
- **No Local Minima in Output Layer**: The linear weights can be solved analytically, guaranteeing a global optimum for that specific layer.

## Cons
- **Curse of Dimensionality**: Performance degrades quickly as the number of input dimensions increases, requiring an exponentially growing number of centers.
- **Center Selection**: The performance is highly sensitive to the method used to choose the centers ($c_j$) and widths ($\sigma_j$).
- **Inference Speed**: Can be slower during evaluation than MLPs if a massive number of centers is required.

## Use Cases
- Function approximation and curve fitting.
- Time series prediction.
- Control systems and signal processing.
- Simple classification tasks.

## Limitations
- Very poor generalization on extremely high-dimensional data (like raw images or audio) compared to deep architectures like CNNs.
- Lacks the deep hierarchical feature extraction capabilities of modern Deep Learning models.
