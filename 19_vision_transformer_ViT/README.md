# Vision Transformer (ViT)

## Detailed Specification
Introduced by Alexey Dosovitskiy et al. from Google Brain in 2020, the Vision Transformer (ViT) demonstrated that a pure Transformer architecture — originally designed for Natural Language Processing — can achieve state-of-the-art results on image classification tasks when trained on sufficient data. The core insight is deceptively simple: an image is split into a sequence of fixed-size patches (e.g., $16\times16$ pixels), each patch is linearly projected into an embedding vector, and the resulting sequence of patch embeddings is fed directly into a standard Transformer encoder. A special learnable `[CLS]` (classification) token is prepended to the sequence, and its final hidden state serves as the image representation for classification. This approach bypasses the inductive biases of CNNs (locality, translation equivariance) entirely, instead relying on the Transformer's self-attention mechanism to learn all spatial relationships from data. When pre-trained on large datasets (ImageNet-21k, JFT-300M), ViT surpasses the best CNNs while requiring substantially fewer computational resources to train.

## Technical Specification
- **Architecture Type**: Pure Transformer Encoder applied to sequences of image patch embeddings.
- **Core Components**:
  - **Patch Embedding**: The input image $x \in \mathbb{R}^{H \times W \times C}$ is divided into a grid of non-overlapping patches of size $P \times P$. Each patch is flattened into a vector of length $P^2 \cdot C$ and linearly projected to dimension $D$ (the model's hidden dimension). This is equivalent to a $P \times P$ convolution with stride $P$.
  - **[CLS] Token**: A learnable embedding vector prepended to the sequence. After passing through the Transformer, the output corresponding to this token is used for classification.
  - **Positional Embeddings**: Learnable 1D positional embeddings are added to each patch embedding (including the [CLS] token) to encode spatial information, since the Transformer has no inherent notion of position.
  - **Transformer Encoder**: A stack of $L$ identical blocks, each containing Multi-Head Self-Attention (MHSA) and a Feed-Forward Network (FFN/MLP), with Layer Normalization applied before each sub-layer (Pre-LN) and residual connections around each.
  - **MLP Head**: The output embedding of the [CLS] token is passed through a small MLP (typically one hidden layer with GELU activation during pre-training, a single linear layer during fine-tuning) to produce the final class logits.

## The Mathematics
- **Patch Embedding:**
  Given an image $x \in \mathbb{R}^{H \times W \times C}$ and patch size $P$, the number of patches is $N = \frac{HW}{P^2}$. Each patch $x_p^i$ is projected:
  $z_0 = [x_{class}; \; x_p^1 E; \; x_p^2 E; \; \ldots; \; x_p^N E] + E_{pos}$
  *(Where $E \in \mathbb{R}^{(P^2 \cdot C) \times D}$ is the linear projection matrix, $x_{class}$ is the [CLS] token, and $E_{pos} \in \mathbb{R}^{(N+1) \times D}$ are the positional embeddings.)*
- **Multi-Head Self-Attention (MHSA):**
  $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$
  *(Each patch embedding attends to every other patch embedding, giving ViT a global receptive field from the very first layer — unlike CNNs where the receptive field grows gradually.)*
- **Transformer Encoder Block:**
  $z'_l = \text{MHSA}(\text{LN}(z_{l-1})) + z_{l-1}$
  $z_l = \text{MLP}(\text{LN}(z'_l)) + z'_l$
  *(Layer Normalization is applied before each sub-layer, and residual connections wrap each sub-layer.)*
- **Classification:**
  $y = \text{MLP}_{head}(\text{LN}(z_L^0))$
  *(Where $z_L^0$ is the final [CLS] token representation after $L$ Transformer layers.)*

## Pros
- **Global Receptive Field**: Every patch can attend to every other patch from the first layer, enabling the model to capture long-range dependencies that CNNs require many layers to achieve.
- **Scalability**: ViT scales remarkably well. Performance improves log-linearly with dataset size and model size, without signs of saturation at the largest scales tested.
- **Unified Architecture**: Uses the exact same Transformer architecture as NLP (BERT, GPT), enabling shared tooling, optimization tricks, and multi-modal extensions (e.g., CLIP, which jointly trains a ViT image encoder with a text Transformer).

## Cons
- **Data Hunger**: Without large-scale pre-training (typically ImageNet-21k or JFT-300M), ViT significantly underperforms comparable CNNs on smaller datasets. The lack of inductive biases (locality, translation equivariance) means these patterns must be learned entirely from data.
- **Quadratic Attention Cost**: Self-attention has $O(N^2)$ complexity in the number of patches. For high-resolution images, $N$ becomes very large, making training and inference expensive without techniques like windowed attention (Swin Transformer).
- **Positional Embedding Rigidity**: The learned positional embeddings are fixed to a specific image resolution. Fine-tuning on a different resolution requires interpolation of the positional embeddings, which can degrade performance.

## Use Cases
- **Large-Scale Image Classification**: Achieves state-of-the-art on ImageNet when pre-trained on JFT-300M.
- **Feature Backbone**: ViT backbones (especially ViT-Large and ViT-Huge) are used as feature extractors in object detection (ViTDet), segmentation (SegFormer, SAM), and generation (DiT).
- **Multi-Modal Learning**: The ViT image encoder is a core component of CLIP (Contrastive Language-Image Pre-training), enabling zero-shot image classification and text-guided image search.
- **Self-Supervised Learning**: ViTs are the architecture of choice for modern self-supervised methods like DINO, DINOv2, and MAE (Masked Autoencoders).

## Limitations
- Training a competitive ViT from scratch on ImageNet-1k alone (without external pre-training data) requires extensive regularization (DeiT recipe: strong augmentation, stochastic depth, repeated augmentation), and still does not match the most optimized CNNs (ConvNeXt) at the same parameter count.
- For real-time inference at the edge (mobile, embedded), the quadratic attention cost and large model size make ViTs impractical without significant compression (distillation, quantization, pruning).
