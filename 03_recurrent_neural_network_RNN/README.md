# Recurrent Neural Network (RNN)

## Detailed Specification
A Recurrent Neural Network (RNN) is a class of artificial neural networks where connections between nodes can create a cycle, allowing output from some nodes to affect subsequent input to the same nodes. This makes them applicable to processing sequences of inputs. Unlike feedforward neural networks, RNNs can use their internal state (memory) to process sequences of inputs.

## Technical Specification
- **Architecture**: Cyclic connections allowing time-dependent data processing.
- **Hidden State**: Serves as the network's memory, capturing information about what has been calculated so far.
- **Backpropagation Through Time (BPTT)**: The algorithm used to train RNNs, unrolling the network through the time sequence.
- **Input/Output Types**: One-to-Many, Many-to-One, Many-to-Many (Seq2Seq).

## The Mathematics
At time step $t$, the RNN receives input $x_t$ and the hidden state from the previous time step $h_{t-1}$.
The new hidden state $h_t$ is computed as:
$$h_t = \tanh(W_{hx} x_t + W_{hh} h_{t-1} + b_h)$$
Where $W_{hx}$ are weights for the input, $W_{hh}$ are weights for the recurrent hidden state, and $b_h$ is the bias.
The output $y_t$ at time step $t$ is computed from the hidden state:
$$y_t = W_{yh} h_t + b_y$$
(Note: An activation function like softmax is typically applied to $y_t$ for classification).

## Pros
- **Handles Sequential Data**: Naturally designed to process data of variable length and temporal nature.
- **Shared Parameters**: Shares weights across all time steps, reducing the number of parameters compared to unrolled networks.
- **Maintains Context**: Theoretical ability to remember information from past inputs.

## Cons
- **Vanishing/Exploding Gradients**: During BPTT, gradients can become extremely small (vanishing) or extremely large (exploding), making it difficult to learn long-range dependencies.
- **Slow Training**: Sequential processing nature prevents parallelization across time steps during training.
- **Short-term Memory**: Standard RNNs struggle in practice to remember information for more than a few time steps.

## Use Cases
- Time series prediction and forecasting.
- Natural Language Processing (NLP) tasks like text classification and sentiment analysis.
- Speech recognition.
- Music generation.

## Limitations
- Fundamentally limited by the vanishing gradient problem, making standard RNNs almost obsolete for complex tasks in favor of LSTMs, GRUs, or Transformers.
- Computational bottleneck due to strictly sequential processing.
