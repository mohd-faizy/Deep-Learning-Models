<div align="center">

# Deep Learning Models Implementation

**A comprehensive notebook repository covering 17 neural-network families in PyTorch, TensorFlow/Keras, and pure-Python/NumPy from-scratch implementations.**

<p>
  <a href="https://github.com/mohd-faizy/Deep-Learning-Models/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License" />
  </a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+" />
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy" />
</p>

<br/>

</div>

---

## Overview

**Deep Learning Models Implementation** is a hands-on curriculum and reference implementation designed to guide you step-by-step through the core architectures of modern deep learning. 

### What You'll Build

- Fundamental networks from scratch using pure Python and NumPy.
- Modern architectures using industry-standard frameworks (PyTorch & TensorFlow).
- Everything from basic MLPs and CNNs to advanced generative models (GANs, Diffusion Models) and graph neural networks.

---

## Learning Outcomes

By exploring this repository, you will be able to:

- Understand the mathematical foundations of 17 distinct neural network architectures.
- Implement forward and backward passes from scratch without relying on autograd.
- Translate theoretical concepts into working code using PyTorch and TensorFlow.
- Compare framework-specific paradigms (e.g., PyTorch's dynamic computational graphs vs. TensorFlow's ecosystem).

---

## Getting Started

### Prerequisites

- Python 3.10+
- Git

### Installation

Clone the repository:

```bash
git clone https://github.com/mohd-faizy/Deep-Learning-Models.git
cd Deep-Learning-Models
```

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Launch Jupyter:

```bash
jupyter lab
```

---

## Tech Stack

| Layer | Tooling |
|:---|:---|
| Frameworks | PyTorch, TensorFlow/Keras |
| Core Math | NumPy |
| Environment | Jupyter Notebooks, Python |

---

## Model Implementations Curriculum

The repository is structured around 17 distinct neural network families.

| # | Architecture | Key Concepts | Directory |
|:---:|:---|:---|:---|
| 01 | **Multilayer Perceptron (MLP)** | Feedforward networks, backpropagation | `01_multilayer_perceptron_MLP` |
| 02 | **Convolutional Neural Network (CNN)** | Convolutions, pooling, feature extraction | `02_convolutional_neural_network_CNN` |
| 03 | **Recurrent Neural Network (RNN)** | Sequential data, hidden states | `03_recurrent_neural_network_RNN` |
| 04 | **Long Short-Term Memory (LSTM)** | Cell states, gating mechanisms | `04_long_short_term_memory_LSTM` |
| 05 | **Gated Recurrent Unit (GRU)** | Simplified gating, update/reset gates | `05_gated_recurrent_unit_GRU` |
| 06 | **Bidirectional RNN** | Past and future context | `06_bidirectional_RNN` |
| 07 | **Transformer & Attention** | Self-attention, multi-head attention | `07_transformer_attention` |
| 08 | **Autoencoder (AE)** | Dimensionality reduction, bottleneck | `08_autoencoder_AE` |
| 09 | **Variational Autoencoder (VAE)** | Probabilistic latent spaces | `09_variational_autoencoder_VAE` |
| 10 | **Generative Adversarial Network (GAN)** | Generator-discriminator min-max game | `10_generative_adversarial_network_GAN` |
| 11 | **Radial Basis Function Network (RBFN)** | Distance-based activation | `11_radial_basis_function_network_RBFN` |
| 12 | **Self-Organizing Map (SOM)** | Unsupervised competitive learning | `12_self_organizing_map_SOM` |
| 13 | **Deep Belief Network (DBN)** | Stacked restricted Boltzmann machines | `13_deep_belief_network_DBN` |
| 14 | **Graph Neural Network (GNN)** | Node embedding, message passing | `14_graph_neural_network_GNN` |
| 15 | **Spiking Neural Network (SNN)** | Biologically inspired leaky integrate-and-fire | `15_spiking_neural_network_SNN` |
| 16 | **Residual Network (ResNet)** | Skip connections, deep training | `16_residual_network_ResNet` |
| 17 | **Diffusion Model (DDPM)** | Forward noise, reverse denoising | `17_diffusion_model_DDPM` |

---

## Repository Structure

```text
Deep-Learning-Models/
|
|-- README.md
|-- requirements.txt
|-- LICENSE
|
|-- 01_multilayer_perceptron_MLP/
|   |-- mlp_pytorch.ipynb
|   |-- mlp_tf.ipynb
|   `-- mlp_from_scratch.py
|
|-- 02_convolutional_neural_network_CNN/
|   |-- cnn_pytorch.ipynb
|   |-- cnn_tf.ipynb
|   `-- cnn_from_scratch.py
|
|-- 03_recurrent_neural_network_RNN/
|   |-- rnn_pytorch.ipynb
|   |-- rnn_tf.ipynb
|   `-- rnn_from_scratch.py
|
|-- ... (and so on for all 17 models)
```

---

## Notes

- Random seeds are set for reproducibility where the framework examples train models.
- Training loops are intentionally short; increase epochs for stronger results.
- The from-scratch scripts focus on core mechanics, not production speed or benchmark accuracy.

---

## Contributing and Support

Contributions are welcome. Please open an issue before submitting major changes.

If this repository helps you, consider giving it a star so other learners can discover it.

<div align="center">
  <br/>
  <p><b>Connect with me</b></p>
  <a href="https://twitter.com/F4izy">
    <img src="https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white" alt="Twitter"/>
  </a>
  <a href="https://www.linkedin.com/in/mohd-faizy/">
    <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn"/>
  </a>
  <a href="https://github.com/mohd-faizy">
    <img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"/>
  </a>
</div>

---

## License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for more information.
