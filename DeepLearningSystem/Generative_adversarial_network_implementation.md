# code
- git clone https://github.com/dlsyscourse/lecture14
- python3 -m pip install pybind11
- 
  ```python
  import sys
  sys.path.append('./python')
  ```

# Components of a generative advesarial network
- There are two main components in a generative adversarial network
  - A generator G that takes a random vector z and maps it to a generated(fake) data G(z).
  - A discriminator that attempts to tell the difference between the real dataset and the fake one.
```
      z ---> Generator_G ---> G(z) ---> Discriminator_D ---> D(G(z)) ---> RoF
  z~N(0,1)    
                              x    ---> Discriminator_D ---> D(z)    ---> RoF
```
  - z: random_vector
  - G(z): fake_data
  - D(G(z)): probablity that input comes from the read dataset
  - RoF: Real or Fake
  - Discriminator_D = $ D(x) = \frac{1}{1+e^{-h(x)}} $


# Parpare the training dataset
- For demonstration purpose, we create our "real" dataset as a two dimensional gaussian distribution.
  - $ X \approx N(u, \sum),\sum = A^T A $
  - $ [3200,2] @ [2,2] = [3200,2] + [2,1] = [3200,2] $
  - Our goal is to create a generator that can generate a distribution that matches this distribution.

# Generator network G
- Now we are ready to build our generator network G, to keep things simple, we make generator an one layer linear neural network.
- At the initialization phase, we just randomly initialized the weight of G, as a result, it certainly does not match the training data. Our goal is to setup a generative adveserial training to get it to close to the training data.

# Discriminator D
- Now let us build a discriminator network  that classifies the real data from the fake one. Here we use a three layer neural network. Additionally, we make use of the Softmax loss to measure the classification likelihood. Because we are only classifying two classes. Softmax function becomes the sigmoid function for prediction.
- $ \frac{exp(x)}{exp(x)+exp(y)} = \frac{1}{1+exp(y-x)} $
- We simply reuse SoftmaxLoss here since this is readily available in our current set of homework iterations. Most implementation will use a binary classification closs instead (BCELoss).
  
# Generative advesarial training
- A Generative adversarial training process iteratively update the generator G and discriminator D to play a "minimax" game.
  - $ min_D max_G \{ -E_{x-Data} logD(x) - E_{z-Noise}log(1-D(G(z))) \} $
- Note that however, in practice, the G update step usually use an alternative objective function.
  - min_G {-E_{z-Noise} log(D(G(z)))}

# Generator update
- Now we are ready to setup the generator update. In the generator update step, we need to optimize the following goal:
  - min_G {-E_{z-Noise} log(D(G(z)))}
- Let us first setup an optimizer for G's parameters.
  ```
    z ---> Generator_G ---> G(z) ---> Oracle_Discriminator ---> D(G(z))   y=1
    z~N(0,1)
  ```
    - z: random_vector
    - G(z): fake data
    - y: label
- To optimize the above loss function, we just need to generate a fake data G(z), send it through the discriminator D and compute the negative log-likelihood that the fake dataset is categorized as real. In another word, we will feed in y=1 as label here.

# Discriminator update
- Now, let us also setup the discriminator update step. The discriminator step optimizes the following objective:
  - $ min_D \{ -E_{x-Data} logD(x) - E_{z-Noise}log(1-D(G(z))) \} $
- Let us first setup an optimizer to learn D's parameters.
  ```
    z ---> Generator_G ---> G(z) ---> Discriminator_D ---> D(G(z)) y=0
  z~N(0,1)    
                            x    ---> Discriminator_D ---> D(x) y=1
  ```
    - z: random_vector
    - G(z): fake data
    - y: label
    - Discriminator_D = D(x) = $ \frac{1}{1+e^{-h(x)}} $

- The discriminator loss is also a normal classification loss, by labeling the generated data as (fake) and real data as (real). Importantly, we also do not need to propagate gradient back to the generator in discriminator update, so we will use the detach function to stop the gradient propagation.

# Putting it together
- Now we can put it together, to summarize, the generative adverserial training cycles through the following steps:
  - The discriminator update step
  - Generator update step
- We can plot the generated data of the trained generator after a number of iterations. As we can see, the generated dataset  after get closer to the real data after training.

# Inspect the trained generator
- We can compare the weight/bias of trained generator G to the parameters we use to genrate the dataset. Importantly, we need to compare the covariance $ \sum = A^T A $ here instead of the transformation matrix.

# Modularizing GAN "Loss"
- We can modularize GAN step as in a similar way as loss function. The following codeblock shows one way to do so.