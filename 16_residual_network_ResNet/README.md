# Residual Network (ResNet)

## Detailed Specification
Residual Network (ResNet) is a groundbreaking deep learning architecture introduced to solve the "degradation problem"—the phenomenon where adding more layers to a deep neural network actually leads to higher training error, not just overfitting. ResNet solves this by introducing "skip connections" or "shortcuts" that allow gradients to bypass certain layers, enabling the successful training of networks with hundreds or even thousands of layers.

## Technical Specification
- **Architecture**: Deep Convolutional Neural Network with Residual Blocks.
- **Core Component**: The Residual Block, containing convolutional layers, batch normalization, and ReLU activations, bypassed by an identity connection.
- **Variants**: ResNet-18, ResNet-34, ResNet-50 (uses bottleneck blocks to save computation), ResNet-101, ResNet-152.

## The Mathematics
In a standard network layer, the goal is to learn an underlying mapping $H(x)$.
In a Residual Block, the network is instead forced to learn the *residual* mapping $F(x)$:
$$F(x) = H(x) - x$$
The output of the block is computed by adding the input $x$ back to the learned residual function $F(x)$ via a skip connection:
$$y = F(x, \{W_i\}) + x$$
If the identity mapping is optimal, it is easier for the network to push the residual $F(x)$ to zero than it is to learn the identity function from scratch using non-linear layers.
During backpropagation, the gradient can flow directly through the identity connection $+x$, mitigating the vanishing gradient problem.

## Pros
- **Extremely Deep Networks**: Allows training of networks with massive depth (100+ layers) without performance degradation.
- **Ease of Optimization**: Solves the vanishing gradient problem effectively.
- **State-of-the-Art Performance**: Became the default backbone for almost all computer vision tasks (object detection, segmentation) upon release.

## Cons
- **Memory Consumption**: Very deep ResNets require significant VRAM to store intermediate activations for backpropagation.
- **Computational Cost**: While bottleneck blocks help, massive ResNets still require heavy compute for training and inference.

## Use Cases
- Image Classification (ImageNet).
- Backbone feature extractor for Object Detection (Faster R-CNN, Mask R-CNN).
- Facial Recognition.
- Medical image analysis.

## Limitations
- If skip connections are used too aggressively without proper regularization, they can bypass too much computation, effectively turning a deep network into a shallow one.
