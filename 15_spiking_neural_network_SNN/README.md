# Spiking Neural Network (SNN)

## Detailed Specification
Spiking Neural Networks (SNNs) are often referred to as the third generation of neural networks. Unlike standard ANNs that use continuous values to represent activations, SNNs more closely mimic biological brains by using discrete "spikes" of electrical activity. Information is encoded not just in the presence of a spike, but in the *timing* and *frequency* of these spikes. SNNs are uniquely suited for deployment on highly specialized neuromorphic hardware.

## Technical Specification
- **Architecture**: Networks of spiking neurons.
- **Neuron Models**: Leaky Integrate-and-Fire (LIF), Izhikevich model, Hodgkin-Huxley model.
- **Information Encoding**: Rate coding, Temporal coding.
- **Learning Rules**: Spike-Timing-Dependent Plasticity (STDP) for unsupervised learning; Surrogate Gradient methods for supervised learning.

## The Mathematics
The most common model is the **Leaky Integrate-and-Fire (LIF)** neuron. The membrane potential $V(t)$ evolves according to a differential equation:
$$\tau_m \frac{dV(t)}{dt} = -(V(t) - V_{rest}) + R \cdot I(t)$$
Where:
- $\tau_m$ is the membrane time constant.
- $V_{rest}$ is the resting potential.
- $I(t)$ is the input current (often the sum of incoming spikes multiplied by synaptic weights).

When the potential reaches a threshold $V_{thresh}$, the neuron emits a spike:
$$V(t) \ge V_{thresh} \implies \text{Spike! and } V(t) \leftarrow V_{reset}$$

## Pros
- **Extreme Energy Efficiency**: On neuromorphic chips (like Intel Loihi or IBM TrueNorth), SNNs consume orders of magnitude less power than CNNs on GPUs, because computations only happen when there's a spike (event-driven).
- **Temporal Processing**: Naturally excellent at handling continuous streams of time-series data without rigid clock steps.
- **Biological Plausibility**: Closest computational model to actual human brain function.

## Cons
- **Non-Differentiable Training**: Spikes are binary (Dirac delta functions), meaning standard backpropagation fails because the derivative is zero almost everywhere. Training requires complex workarounds like "surrogate gradients."
- **Lack of Hardware**: While software simulations exist, their true efficiency benefits require specialized, rare neuromorphic hardware.
- **Accuracy Gap**: SNNs generally trail behind deep CNNs/Transformers in pure accuracy on standard benchmark tasks.

## Use Cases
- Ultra-low power edge AI devices (IoT sensors, smart prosthetics).
- Brain-Computer Interfaces (BCI).
- Event-based camera processing (DVS cameras).
- Robotics.

## Limitations
- Difficult to scale to the billions of parameters seen in modern LLMs.
- The ecosystem (frameworks like snnTorch, Nengo) is still maturing compared to PyTorch/TensorFlow.
