
# Transformers and Attention

# Outline
- The two approaches to time series modeling
- Self-attention and Transformers
- Transformers beyond time series (very briefly)

# The two approaches to time series modeling

## Time series prediction

- Let’s recall our basic time series prediction task from the previous lectures
  ```
      y_1     y_2     y_3
      /|\     /|\     /|\
       |       |       |
      x_1 ->  x_2 ->  x_3 -> ...
  ```
- More fundamentally, a time series prediction task is the task of predicting
  $ y_{1:T} = f_{\theta}(x_{1:T}) $
- where y_t can depend only on x_{1:t}
- There are multiple methods for doing so, which may or may not involve the latent state representation of RNNs

## The RNN “latent state” approach

  ```
            y_1     y_2     y_3
            /|\     /|\     /|\
             |       |       |
    h_0 ->  h_1  -> h_2  -> h_3 -> ... 
            /|\     /|\     /|\
             |       |       |
            x_1     x_2     x_3
  ```
- We have already seen the RNN approach to time series: maintain “latent state” h_t that summarizes all information up until that point
- Pros: Potentially “infinite” history, compact representation
- Cons: Long “compute path” between history and current time ⟹ vanishing / exploding gradients, hard to learn


## The “direct prediction” approach
- In contrast, can also directly predict output $ y_t = f_{\theta}(x_{1:t}) $
- (just need a function that can make predictions of differently-sized inputs)
- Pros: Often can map from past to current state with shorter compute path
- Cons: No compact state representation, finite history in practice

## CNNs for direct prediction

  ```
            y_1    y_2     y_3     y_4     y_5
            /|\  //\/|\  //\/|\  //\/|\  //\/|\
             |   /   |   /   |   /   |  /   |

            ...    ...     ...     ...     ...  
            /|\  //\/|\  //\/|\  //\/|\  //\/|\
             |   /   |   /   |   /   |  /   |
           z_1^2   z_2^2   z_3^2   z_4^2   z_5^2
            /|\  //\/|\  //\/|\  //\/|\  //\/|\
             |   /   |   /   |   /   |  /   |
           z_1^1   z_2^1   z_3^1   z_4^1   z_5^1
            /|\  //\/|\  //\/|\  //\/|\  //\/|\
             |   /   |   /   |   /   |  /   |
              x_1     x_2     x_3     x_4   x_5
  ```
- One of the most straightforward ways to specify the function f_{\theta}: (fully) convolutional networks, a.k.a. temporal convolutional networks (TCNs)
- The main constraint is that the convolutions be causal: $ z_t^{i+1} $ can only depend on $ z_{t-k:t}^{(i)} $
- Many successful applications: e.g. WaveNet for speech generation (van den Oord et al.,2016)
  
## Challenges with CNNs for dense prediction
- Despite their simplicity, CNNs have a notable disadvantage for time series prediction: the receptive field of each convolution is usually relatively small ⟹ need deep networks to actually incorporate past information
- Potential solutions:
  - Increase kernel size: also increases the parameters of the network
  - Pooling layers: not as well suited to dense prediction, where we want to predict all of $ y_{1:T} $
  - Dilated convolutions: “Skips over” some past state / inputs

---

# Self-attention and Transformers


  ```
            w_1     w_2     w_3
            /|\     /|\     /|\
             |       |       |
  h_0^k -> h_1^k -> h_2^k -> h_3^k -> ... 
            /|\     /|\     /|\
             |       |       |

            ...     ...     ...
            /|\     /|\     /|\
             |       |       |
  h_0^1 -> h_1^1 -> h_2^1 -> h_3^1 -> ... 
            /|\     /|\     /|\
             |       |       |
            x_1     x_2     x_3
  ```

## “Attention” in deep learning
- “Attention” in deep networks generally refers to any mechanism where individual states are weighted and then combined
- $ z_t = \theta^T h_t^{(k)} $
- $ w = softmax(z) $
- $ \overline h = \sum_{t=1}^T w_t h_t^{(k)} $
- Used originally in RNNs when one wanted to combine latent states over all times in a more general manner than “just” looking at the last state

## The self-attention operation

- Self-attention refers to a particular form of attention mechanism
- Given three inputs $ K,Q,V \in R^{T,d} $ (“keys”, “queries”, “values”, in one of the least-meaningful semantic designations we have in deep learning)
  - $ K = 
      \begin{bmatrix}
      \sim & k_1^{T} & \sim \\
      \sim & k_2^{T} & \sim\\
      \sim & ... & \sim \\
      \sim & k_T^{T} & \sim \\
      \end{bmatrix} $

  - $ Q = \begin{bmatrix}
      \sim & q_1^{T} & \sim \\
      \sim & q_2^{T} & \sim\\
      \sim & ... & \sim \\
      \sim & q_T^{T} & \sim \\
      \end{bmatrix} $

  - $ V = \begin{bmatrix}
      \sim & v_1^{T} & \sim \\
      \sim & v_2^{T} & \sim\\
      \sim & ... & \sim \\
      \sim & v_T^{T} & \sim \\
      \end{bmatrix} $

- we define the self attention operation as
  - $ SelfAttention(K,Q,V) = softmax(\frac{KQ^T}{d^{\frac12}})V $

## Self-attention in more detail
- $ softmax(\frac{KQ^T}{d^{\frac12}})V $
  - softmax: applied rowwise
  - $ \frac{KQ^T}{d^{\frac12}} $ is TxT "weight" matrix
- Properties of self-attention
  - Invariant (really, equivariant) to permutations of the K,Q,V matrices
  - Allows influence between $ k_t, q_t, v_t $ over all times
  - Compute cost is $ O(T^2 + Td) $ (cannot be easily reduced due to nonlinearity applied to full T×T matrix)

## Transformers for time series
- The Transformer architecture uses a series of attention mechanisms (and feedfoward layers) to process a time series
  - $ Z^{i+1} = Transformer(Z^i) $
  - (described in detail on next slide)
- All time steps (in practice, within a given time slice) are processed in parallel, avoids the need for sequential processing as in RNNs

## Transformer block
```
        output
         /|\
          |
       normalize
       \ /|\
  |------ |
  |    /  |
  |  feed forward
  |      /|\
  |_______|
          |
       normalize
       \ /|\
  |------ |
  |    /  |
  |    multiply
  |      /|\
  |   ---------
  |   |       |
  | softmax   |
  |  /|\      |
  |   |       |
  | multiply  |
  |  /|\      |
  |   |       |
  |  ----     |
  |  |  |     |
  |  Q  k     V
  | /|\/|\   /|\
  |  |__|_____|
  |       |
  |_____input
```

- In more detail, the Transformer block has the following form:
  - $ \hat Z := SelfAttention(Z^{(i)}W_K, Z^{(i)}W_Q, Z^{(i)}W_V) = softmax(\frac{Z^{(i)}W_K(Z^{(i)}W_V)^T}{d^{\frac12}})Z^{(i)}W_V  $
  - $ = softmax(\frac{Z^{(i)}W_K W_V^T(Z^{(i)})^T}{d^{\frac12}})Z^{(i)}W_V $
  - $ \hat Z := LayerNorm(Z^{(i)} + \hat Z ) $
  - $ Z^{(i+1)} := LayerNorm(ReLU(\hat Z W) + \hat Z) $
- A bit complex, but really just self-attention, followed by a linear layer + ReLU, with residual connections and normalization thrown in somewhat arbitrarily (and precisely where these go are often adjusted)


## Transformers applied to time series
- We can apply the Transformer block to the “direct” prediction method for time series, instead of using a convolutional block
- Pros:
 - Full receptive field within a single layer (i.e., can immediately use past data)
 - Mixing over time doesn’t increase parameter count (unlike convolutions)
Cons:
 - All outputs depend on all inputs (no good e.g., for autoregressive tasks)
 - No ordering of data (remember that transformers are equivariant to permutations of the sequence)

## Masked self-attention
- To solve the problem of “acausal” dependencies, we can mask the softmax operator to assign zero weight to any “future” time steps
  - $ softmax(\frac{KQ^T}{d^{\frac12}} - M)V  , M = \begin{bmatrix}0 & ... & \infty & \infty \\ 0 & ... & 0 & \infty\\ 0 & ... & 0 & 0\\ \end{bmatrix} $
- Note that even though technically this means we can “avoid” creating those entries in the attention matrix to being with, in practice it’s often faster to just form them then mask them out (more on Monday)

## Positional encodings
- To solve the problem of “order invariance”, we can add a <font color=red>positional encoding</font> to the input, which associates each input with its position in the sequence
- $ X \in R^n = 
    \begin{bmatrix}
     \sim & x_1^T & \sim \\ 
     \sim & x_2^T & \sim \\
     \sim & ... & \sim \\
     \sim & x_T^T & \sim \\ 
    \end{bmatrix}
    +
    \begin{bmatrix}
    sin(w_1.1) & ... & sin(w_n.1) \\
    sin(w_1.2) & ... & sin(w_n.2) \\
    & ... & \\
    sin(w_1.T) & ... & sin(w_n.T) \\
    \end{bmatrix} $
  - and where $ w_i,i = 1, … n $ is typically chosen according to a logarithmic schedule (Really, add positional encoding to d-dimensional projection of X)

# Transformers beyond time series (very briefly)

## Transformers beyond time series
- Recent work has observed that transformer blocks are extremely powerful beyond just time series
 - Vision Transformers: Apply transformer to image (represented by a collection of patch embeddings), works better than CNNs for large data sets
 - Graph Transformers: Capture graph structure in the attention matrix
- In all cases, key challenges are:
 - How to represent data such that $ O(T^2) $ operations are feasible
 - How to form positional embeddings
 - How to form the mask matrix

---

# Implementing Transformers
- This notebook will walk you through the internals of implementing self attention and transformer networks. As with recurrent networks (and unlike convolutions), there is actually relatively little that is fundamentally new in their implementation, as it all largely involves an application of existing primitives you will have already implemented in your autodiff framework. However, there is indeed one aspect of an efficient implementation that requires a slight generalization of an item we have discussed already: a batch version of matrix multiplication. This is required for both the minibatch version of attention as well as the common "multihead" version. We will also briefly discuss some approaches to making Transformers more efficient.

## Implementing self-attention
- Let's begin with a simple implementation of self-attention. This essentially just implements the basic equation
  - $ Y = softmax(\frac{KQ^T}{d^{\frac12}})V $
- By convention, however, it's typical to implement self attention in terms of the actual inputs X rather than the K, Q, and V values themselves (i.e., instead of having the linear layer separately). It's also common to have an output weight as well (even though this could in theory be folded into the  $W_{KQV}$ terms), which applies an additional linear layer to the output of the the entire operation. I.e., the full operation is given by
  - $ Y = (softmax(\frac{XW_K W_Q^T X^T}{d^{\frac12}})XW_v)W_o $
- It's possible to also incorporate bias terms into each of these projections, though we won't bother with this, as it is less common for everything but the output weight, and then just largely adds complexity.
- Let's see what this implementation looks like.

- We can compare this to PyTorch's self-attention implementation, the nn.MultiheadAttention layer (we'll cover what we mean by "multi-head" shortly). Note that by default (mainly just to be similar to the RNN implementation and other sequence models, the nn.MultiheadAttention layer also by default takes inputs in (T,N,d) form (i.e, the batch dimension second. But unlike for RNNs, this ordering doesn't make much sense for self-attention and Transformers: we will be computing the operation "in parallel" over all times points, instead of as a sequential model like for RNNs. So we'll use the batch_first=True flag to make this a more natural dimension ordering for the inputs.


# Minibatching with batch matrix multiply

- Once we move from single example to minibatches, there is one additional subtlety that comes into play for self-attenion. Recall that for each sample in the minibatch, we will have to compute a matrix product, e.g., the $ KQ^T $ term. If we need to process examples in a minibatch, we will need to perform this matrix multiplication correspondingly for each sample. This is an operation known as a batch matrix multiply.

- It may seem as though nothing is new here. True, for an MLP it was possible to perform the entire batch equation as a single matrix multiplication, but didn't we similarly need to batch matrix multiplications for convolutional networks (after the im2col function)? Or for RNNs?

- The answer is actually that no, previous to this we haven't needed the true batch matrix multiplication fuctionality. The situations we had before involved the multiplication of a "batched" tensor by a single weight matrix. I.e., in a ConvNet, we had something like
  - $ y = im2col(x)W $
- or in the batched setting
  - $ y^{(i)} = im2col(x^{(i)})W $

- But this operation can be accomplished with "normal" matrix multiplication by just stacking the multiple samples into the matrix on the left
  - $ \begin{bmatrix}
      y^{(1)} \\
      y^{(2)} \\
      ... \\
      y^{(N)} \\
    \end{bmatrix} = \begin{bmatrix}
      im2col(x^{(1)}) \\
      im2col(x^{(2)}) \\
      ... \\
      im2col(x^{(N)}) \\
    \end{bmatrix}
    W
    $
- This operation is just a normal matrix multiplication, so can be implemented e.g., using your framework so far, where matrix multiplication always operates on 2 dimensional NDArrays.

- Fortunately, numpy's @ operator already performs batch matrix multiplication for the case of multiple arrays of (the same) dimension more than 2.

- Let's see how this works with our self attention layer. In fact, because of the judicious usage of axis=-1 and similar terms, our layer works exactly the same as it did before.


# Multihead attention
- Practical implementations of attention use what is called multihead attention, which simply means that we run the self-attention mechansism of different subsets of the K, Q, V  terms, then concatenate them together. Formally, we'll partition these terms as
  - $ K = [K_1 \ K_2 \ ... \ K_{heads} ] $
(and similarly for Q and V.
  - $ Q = [Q_1 \ Q_2 \ ... \ Q_{heads} ] $
  - $ V = [V_1 \ V_2 \ ... \ V_{heads} ] $
- Then will form the self attention outputs
  - $ Y_i = softmax(\frac{K_iQ_i^T}{(\frac{d}{heads})^{\frac12}})V_i $
- and then form the final ouput
  - $ Y = [Y_1 \ Y_2 \ ... \ Y_{heads} ]W_o $
- The advantage of multi-head attention is that applying a single self-attention layer to a "high dimensional" hidden state (i.e., where  is large) seems to waste a lot of the information contained in the hidden layers. Recall, for intance, that the terms in the self attention matrix would be proportation to $ k_t^T q_s $. If $ k_t $
 and $ q_s $ are high dimensional, then a lot of "internal structure" could be lost to result in ultimately just one weighting term. By breaking this up and computing multiple differen attention matrices, each of which weights different dimensions of the V term, we avoid this problem, and practically lead to better performance. Note however that the "right" tradeoff between the number of heads and  is still rather heuristic in nature.

# Transformer Block
- Let's finally put all this together into a full transformer block. Transformers simply amount to a self-attention block, with a residual layers and layer norm operation, followed by a two-layer feedforward network, with another residual layer and layer norm. We can implement this in a few lines of code. Note that in "real" implementations, the layer norm terms, etc, would actually have trainable scale/bias terms that add a bit more expressivity to the model. This version we show will only be the same, for instance, at initialization.

# The question for "efficient Transformers"
- Since the Transformer was first proposed, there have been endless attempts made to make different "efficient" versions of the operation. The key drawback of transformers, we have seen, is that they require forming a the TxT attention matrix and multiplying by V (an $O(T^2d)$ operation)
  - $ softmax(\frac{KQ^T}{d^{\frac12}}) V $
- If T is much larger than d (e.g., the sequence is very long, then this operation is quite costly).
- There are essentially two approaches to making the approach more efficient: by attempting the represent the attention matrix
  - $ A = softmax(\frac{KQ^T}{d^{\frac12}}) $

- either using sparsity or using low rank structure. In general, of course, this matrix neither sparse nor low rank. But we could simply dicate, for example, that we will only compute some subset of the attention weights, thereby decreasing the number of inner products we need to perform (this is the basis of the so-called "Sparse Attention" layer: similar approaches have been proposed a number of times, but this is one such example). Alternatively, one could try to infer some kind of hard sparsity by e.g., triangle inequalities or other similar instances (because, remember, we are computing what amounts to a similarly metric between the x terms at different times).

- Alternatively, we could try to represent A in low rank form instead. To see why this could be appealing, consider the case where we don't have a softmax operation at all, but instead used the "attention" layer
  - $ (\frac{KQ^T}{d^{\frac12}})V $
- In this case, if T >> d, we could instead perform our multiplication in the order $K{Q^T}V$, which would only have complexity $O(Td^2)$, potentially much smaller. And some papers infact advocate for this very thing, or alternatively try to find a low-rank representation of the actual attention weights, to similar effects.

- The thing to keep in mind with all these "efficient" alternatives (and if you have been reading the literation surrounding Transformers, you have likely seen a ton of these), is whether they are actually more efficient, for an equivalent level of performance, once real execution speed in taken into account. My best understanding of the current situation is that 1) explicit sparse self attention is indeed sometimes useful for models that want very long history, but that 2) most of the "efficient" transformer mechanisms that use low rank structure or inferred sparsity structure don't improve much in practice over traditional attention.