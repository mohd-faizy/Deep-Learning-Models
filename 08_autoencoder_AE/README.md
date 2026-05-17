# Autoencoder (AE)

## Detailed Specification
An Autoencoder is a type of artificial neural network used to learn efficient codings of unlabeled data (unsupervised learning). The objective of an autoencoder is to learn a representation (encoding) for a set of data, typically for dimensionality reduction, by training the network to ignore signal "noise." Along with the reduction side, a reconstructing side is learned, where the autoencoder tries to generate from the reduced encoding a representation as close as possible to its original input.

## Technical Specification
- **Architecture**: Symmetrical Encoder-Decoder structure with a central bottleneck.
- **Encoder**: Compresses the high-dimensional input $x$ into a low-dimensional latent space representation $h$.
- **Bottleneck**: The layer with the smallest dimensions, forcing the network to learn a compressed representation.
- **Decoder**: Reconstructs the input $\hat{x}$ from the latent space $h$.
- **Variants**: Denoising Autoencoder, Sparse Autoencoder, Contractive Autoencoder.

## The Mathematics
Let the input be $x$.
The **Encoder** function $f$ maps $x$ to latent space $h$:
$$h = f(x) = \sigma(Wx + b)$$
The **Decoder** function $g$ maps $h$ back to the original space $\hat{x}$:
$$\hat{x} = g(h) = \sigma'(W'h + b')$$
The **Loss Function** is typically the Mean Squared Error (MSE) or Binary Cross-Entropy (BCE) between the input and the reconstruction:
$$L(x, \hat{x}) = ||x - g(f(x))||^2$$

## Pros
- **Unsupervised Learning**: Does not require labeled data.
- **Non-linear Dimensionality Reduction**: Can learn more complex projections than linear methods like PCA.
- **Feature Extraction**: The learned latent representations can be used as inputs for other supervised learning tasks.

## Cons
- **Identity Function Risk**: If the bottleneck is too wide or the network is too powerful, it might simply learn to copy the input directly without extracting meaningful features.
- **Not Generative**: Standard Autoencoders do not learn a continuous, meaningful probability distribution in the latent space, making them poor choices for generating *new* data.
- **Data Specificity**: Autoencoders are highly specific to the training data.

## Use Cases
- Dimensionality Reduction.
- Image Denoising (Denoising Autoencoders).
- Anomaly Detection (anomalies will have high reconstruction error).
- Image Compression.

## Limitations
- Standard AEs map discrete points in latent space, resulting in "gaps." Sampling from these gaps yields meaningless outputs.
- Interpolation in the latent space is often not smooth.
