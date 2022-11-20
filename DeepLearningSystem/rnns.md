# Sequence Modeling and Recurrent Networks

# Outline
- Sequence modeling
- Recurrent neural networks
- LSTMs
- Beyond "simple" sequential models

# Sequence modeling

## Sequence modeling tasks

- In the examples we have considered so far, we make predictions assuming each
input output pair x(i), y(i) is independent identically distributed (i.i.d.)
- In practice, many cases where the input/output pairs are given in a specific
sequence, and we need to use the information about this sequence to help us
make predictions
  ```
    y(1) -> y(2) -> y(3)
    /|\     /|\     /|\
     |       |       |
    x(1) -> x(2) -> x(3) -> ... 
  ```
  or more commonly, denoting x_t as a whole vector
  ```
    y_1  -> y_2  -> y_3
    /|\     /|\     /|\
     |       |       |
    x_1  -> x_2  -> x_3 -> ... 
  ```

## Example: Part of speech tagging

- Given a sequence of words, determine the part of speech of each word
  ```
     DT       JJ       JJ       NN     VBD
    /|\      /|\      /|\      /|\     /|\
     |        |        |        |       |
    The  -> quick  -> brown -> fox -> jumped 
  ```
  - A word’s part of speech depends on the context in which it is being used, not just on the word itself

## Example: speech to text

- Given a audio signal (assume we even know the word boundaries, and map each segment to a fix-sized vector descriptor), determine the corresponding transcription
  ![](pictures/speech_to_text.png)
  - Again, context of the words is extremely important (see e.g., any bad speech recognition system that attempts to “wreck a nice beach”)

## Example: autoregressive prediction
- A special case of sequential prediction where the elements to predict is the next element in the sequence
  ```
     quick  brown     fox     jumped   over
    /|\      /|\      /|\      /|\     /|\
     |        |        |        |       |
    The  -> quick  -> brown -> fox -> jumped 
  ```
  - Common e.g., in time series forecasting, language modeling, and other use cases

# Recurrent neural networks

## Recurrent neural networks

- Recurrent neural networks (RNNs) maintain a hidden state over time, which is a function of the current input and previous hidden state
  ```
            y_1     y_2     y_3
            /|\     /|\     /|\
             |       |       |
    h_0 ->  h_1  -> h_2  -> h_3 -> ... 
            /|\     /|\     /|\
             |       |       |
            x_1     x_2     x_3
  ```
  - $ h_t = f(W_{hh}h_{t-1} + W_{hx}x_t + b_h) $
  - $ y_t = g(W_{yh}h_t + b_y) $
    - where f and g are activation functions
    - W_{hh}, W_{hx}, W_{yh} are weights and 
    - b_h, b_y are bias terms

## How to train your RNN
- Given a sequence of inputs and target outputs (x_1,...,x_T,y*_1,...,y*_T), we can train an RNN using backpropagation through time, which just involves “unrolling” the RNN over the length of the sequence, then relying mostly on autodiff
  ```
    opt = Optimizer(params = (W_hh, W_hx, W_yh, b_h, b_y))
    h[0] = 0
    l = 0
    for t = 1,...,T:
      h[t] = f(W_hh * h[t-1] + W_hx*x[t] + b_h)
      y[t] = g(W_yh * h[t] + b_y)
      l += Loss(y[t], y_star[t])
    l.backward()
    opt.step()
  ```

## Stacking RNNs
- Just like normal neural networks, RNNs can be stacked together, treating the hidden unit of one layer as the input to the next layer, to form “deep” RNNs
- Practically speaking, tends to be less value in “very deep” RNNs than for other architectures

  ```
            y_1      y_2      y_3
            /|\      /|\      /|\
             |        |        |
            ...      ...      ...
            /|\      /|\      /|\
             |        |        |
    h2_0 -> h2_1  -> h2_2  -> h2_3 -> ... 
            /|\      /|\      /|\
             |        |        |
    h1_0 -> h1_1  -> h1_2  -> h1_3 -> ... 
            /|\      /|\      /|\
             |        |        |
            x_1      x_2      x_3
  ```

## Exploding activations/gradients
- The challenge for training RNNs is similar to that of training deep MLP networks
- Because we train RNNs on long sequences, if the weights/activation of the RNN are scaled poorly, the hidden activations (and therefore also the gradients) will grow unboundedly with sequence length
- Single layer RNN with ReLU activations, using weight initialization
  - $ W_{hh} \approx N(0,3/n) $
  - Recall that $ \sigma^2 = 2/n $ was the “proper” initialization for ReLU activations

## Vanishing activation/gradients
- Similarly, if weights are too small then information from the inputs will quickly decay with time (and it is precisely the “long range” dependencies that we would often like to model with sequence models)
- Single layer RNN with ReLU activations, using weight initialization
- $ W_{hh} \approx N(0,1.5/n) $
- Non-zero input only provided here for time 1, showing decay of information about this input over time

# Alternative Activations
- One obvious problem with the ReLU is that it can grow unboundedly; does using bounded activations “fix” this problem?
- $ sigmoid(x) = \frac{1}{1+e^{-x}} $
- $ tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} $
- No … creating large enough weights to not cause activations/gradients to vanish requires being in the “saturating” regions of the activations, where gradients are very small ⟹ still have vanishing gradients
  
# LSTMs

## Long short term memory RNNs
- Long short term memory (LSTM) cells are a particular form of hidden unit update that avoids (some of) the problems of vanilla LSTMs
- Step 1: Divide the hidden unit into two components, called (confusingly) the hidden state and the cell state

  ```
            y_1     y_2     y_3
            /|\     /|\     /|\
             |       |       |
    h_0 ->  h_1  -> h_2  -> h_3 -> ... 
            /|\     /|\     /|\
             |       |       |
            x_1     x_2     x_3
  ```
  -->
  ```
            y_1     y_2     y_3
            /|\     /|\     /|\
             |       |       |
    h_0     h_1     h_2     h_3
    ... ->  ...  -> ...  -> ... -> ...
    c_0     c_1     c_2     c_3
            /|\     /|\     /|\
             |       |       |
            x_1     x_2     x_3
  ```

## Long short term memory RNNs
- Step 2: Use a very specific formula to update the hidden state and cell state (throwing in some other names, like “forget gate”, “input gate”, “output gate” for good measure)

- $ \begin{bmatrix}
    i_t \\
    f_t \\
    g_t \\
    o_t \\
   \end{bmatrix} = 
   \begin{bmatrix}
    sigmoid \\
    sigmoid \\
    tanh \\
    sigmoid \\
   \end{bmatrix}* (W_{hh}h_{t-1} + W_{hx}x_t + b_h)
  $
  - $ c_t = c_{t-1} * f_t + i_t * g_t $
  - $ h_t = tanh(c_t) * o_t $

## Why do LSTMs work?
- There have been a seemingly infinite number of papers / blog posts about “understanding how LSTMs work” (I find most of them rather unhelpful)
- The key is this line here:
  - $ c_t = c_{t-1} * f_t + i_t * g_t $
  - We form $ c_t $ by scaling down $ c_{t-1} $ (remember, $ f_t $ is in $ [0,1]^n $, then adding a term to it
  - Importantly, “saturating” sigmoid activation for $ f_t $ at 1 would just pass through $ c_{t-1} $ untouched
  - ⟹ For a wide(r) range of weights,LSTMs don’t suffer vanishing gradients

## Some famous LSTMs
- A notably famous blog post in the history of LSTMs:
- http://karpathy.github.io/2015/05/21/rnn-effectiveness/
- Andrej’s blog post

```
/*
* Increment the size file of the new incorrect
UI_FILTER group information
* of the size generatively.
*/
static int indicate_policy(void)
{
int error;
if (fd == MARN_EPT) {
/*
* The kernel blank will coeld it to userspace.
*/
if (ss->segment < mem_total)
unblock_graph_and_set_blocked();
else
ret = 1;
goto bail;
}
segaddr = in_SB(in.addr);
selector = seg / 16;
setup_works = true;
…
```
- Generation from character-by-character autoregressive model trained on Linux source code
- … trained on Latex source code
  
# Beyond "simple" sequential models

## Sequence-to-sequence models
- To give you a short glimpse of the kind of things you can do with RNNs/LSTMs beyond “simple” sequence prediction, consider the task of trying to translate between languages
- Can concatenate two RNNs together, one that “only” processes the sequence to create a final hidden state (i.e., no loss function); then a section that takes in this initial hidden state, and “only” generates a sequence

  ```
    zhe      kuai     zong     huli    <STOP>
    /|\      /|\      /|\      /|\     /|\
     |        |        |        |       |       
    h6  ->    h7  ->   h8   ->  h9 ->  h10
    |
    -------------------------------------
                                        |
     h1   ->  h2  ->   h3  ->  h4  ->  h5
    /|\      /|\      /|\      /|\     /|\
     |        |        |        |       |
    The    quick     brown    fox     <STOP> 
  ```

## Bidirectional RNNs
- RNNs can use only the sequence information up until time t to predict $ y_t $
- - This is sometimes desirable (e.g., autoregressive models)
  - But sometime undesirable (e.g., language translation where we want to use “whole” input sequence)
- Bi-directional RNNs: stack a forwardrunning RNN with a backward-running RNN: information from the entire sequence to propagates to the hidden state

  ```
            y_1     y_2     y_3
            /|\     /|\     /|\
             |       |       |
            ...     ...     ...            
            /|\     /|\     /|\
             |       |       |
            h2_1 <- h2_2 <- h2_3 <- ... <- h2_{T+1}
            /|\     /|\     /|\
             |       |       |
    h1_0 -> h1_1 -> h1_2 -> h1_3 -> ...
            /|\     /|\     /|\
             |       |       |
            x_1     x_2     x_3
  ```