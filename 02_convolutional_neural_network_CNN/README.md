# Convolutional Neural Network (CNN)

## Detailed Specification
A Convolutional Neural Network (CNN) is a class of deep neural networks commonly applied to analyzing visual imagery. They are also known as shift invariant or space invariant artificial neural networks (SIANN), based on the shared-weight architecture of the convolution kernels or filters that slide along input features and provide translation-equivariant responses known as feature maps.

## Technical Specification
- **Architecture**: Feedforward with specialized convolutional and pooling layers.
- **Key Operations**: Convolution (filtering), Pooling (downsampling), Flattening, Fully Connected (Dense).
- **Filters/Kernels**: Small matrices that slide over the input data to extract features like edges, textures, and shapes.
- **Pooling Mechanisms**: Max Pooling, Average Pooling.
- **Common Architectures**: LeNet, AlexNet, VGG, Inception, ResNet.

## The Mathematics
The core operation is the discrete convolution. For a two-dimensional image $I$ and a two-dimensional kernel $K$, the convolution operation is defined as:
$$S(i,j) = (I * K)(i,j) = \sum_m \sum_n I(i+m, j+n) K(m,n)$$
Where $S(i,j)$ is the resulting feature map at position $(i,j)$.
Activation functions (like ReLU) are applied element-wise: $A(i,j) = \max(0, S(i,j))$.
Pooling layers then reduce the spatial dimensions, e.g., Max Pooling with a $2\times2$ window:
$$P(i,j) = \max_{m,n \in \{0,1\}} A(2i+m, 2j+n)$$

## Pros
- **Parameter Sharing**: A single filter is used across the entire image, drastically reducing the number of parameters compared to MLPs.
- **Spatial Hierarchy**: Naturally builds hierarchical representations (edges $\rightarrow$ shapes $\rightarrow$ objects).
- **Translation Invariance**: Pooling layers help the network recognize objects regardless of their exact position in the image.

## Cons
- **Computationally Intensive**: Convolution operations require significant computational power, often necessitating GPUs.
- **Need for Large Datasets**: CNNs typically require massive amounts of labeled data to train from scratch without overfitting.
- **Vulnerability to Adversarial Attacks**: Small, imperceptible changes to an image can completely change the network's prediction.

## Use Cases
- Image classification and categorization.
- Object detection and bounding box prediction (YOLO, Faster R-CNN).
- Image segmentation (U-Net, Mask R-CNN).
- Facial recognition.
- Medical image analysis.

## Limitations
- Lacks understanding of spatial relations beyond the receptive field (e.g., might recognize eyes, nose, mouth, but not care if they are arranged correctly).
- Not naturally suited for sequential or non-grid-like structured data (like graphs or text, though 1D CNNs exist for sequences).
