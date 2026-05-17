# Self-Organizing Map (SOM)

## Detailed Specification
A Self-Organizing Map (SOM), or Kohonen map, is a type of artificial neural network trained using unsupervised learning to produce a low-dimensional (typically two-dimensional), discretized representation of the input space of the training samples, called a map. SOMs differ from other artificial neural networks as they apply competitive learning as opposed to error-correction learning (like backpropagation), and in the sense that they use a neighborhood function to preserve the topological properties of the input space.

## Technical Specification
- **Architecture**: Grid of neurons (usually 2D). Every input is connected to every neuron in the grid.
- **Learning Type**: Unsupervised, Competitive Learning.
- **Key Concepts**:
  - **Best Matching Unit (BMU)**: The neuron whose weight vector is closest to the input vector.
  - **Neighborhood Function**: Determines how heavily the weights of the BMU's neighbors are adjusted.
  - **Decay**: Learning rate and neighborhood radius decay over time.

## The Mathematics
Given an input vector $x(t)$ at epoch $t$:
1. **Find BMU**: Calculate Euclidean distance to all weight vectors $W_v$. The BMU $u$ is:
   $$u = \arg\min_v ||x(t) - W_v(t)||$$
2. **Weight Update**: Adjust the weights of the BMU and its neighbors:
   $$W_v(t+1) = W_v(t) + \theta(u, v, t) \cdot \alpha(t) \cdot (x(t) - W_v(t))$$
   Where:
   - $\alpha(t)$ is a monotonically decreasing learning rate.
   - $\theta(u, v, t)$ is the neighborhood function, typically a Gaussian centered on the BMU $u$, which shrinks over time:
     $$\theta(u, v, t) = \exp\left(-\frac{||r_u - r_v||^2}{2\sigma^2(t)}\right)$$

## Pros
- **Topological Preservation**: Similar data points are mapped to adjacent neurons, making it excellent for visualization of complex data.
- **Unsupervised**: Discovers underlying structures in data without requiring labels.
- **Intuitive Visualizations**: The resulting 2D grid (U-Matrix) is highly interpretable for humans.

## Cons
- **Hyperparameter Sensitivity**: Highly sensitive to the initial initialization, learning rate schedule, neighborhood radius, and the chosen map dimensions.
- **Requires Sufficient Data**: Needs a large and representative dataset to form meaningful clusters.
- **Lack of Cost Function**: Does not have a clear objective function being minimized, making it hard to quantitatively evaluate convergence or compare models.

## Use Cases
- Dimensionality reduction and data visualization.
- Clustering and identifying natural groupings in data.
- Color quantization.
- Feature extraction prior to supervised learning.

## Limitations
- Determining the exact optimal grid size (e.g., 10x10 vs 20x20) is mostly heuristic.
- Struggles with categorical data, as Euclidean distance metrics are assumed.
