import struct
import gzip
import numpy as np

import sys
sys.path.append('python/')
import needle as ndl


def parse_mnist(image_filesname, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.
    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format
    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.
            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    with gzip.open(image_filesname, 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        image = np.frombuffer(f.read(), dtype=np.dtype(np.uint8).newbyteorder('>'))
        image = image.reshape((size, nrows*ncols))
    # In normalization, we map the minimum feature value to 0 and the maximum to 1. 
    # Hence, the feature values are mapped into the [0, 1] range:
    min = image.min()
    max = image.max()
    normalize_image = (np.float32)(image - min)/(max - min)

    with gzip.open(label_filename,'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        label = np.frombuffer(f.read(), dtype=np.dtype(np.uint8).newbyteorder('>'))
        label = label.reshape((size,)) # (Optional)
    return normalize_image, label
    ### END YOUR CODE


def softmax_loss(Z, y_one_hot):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.
    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.
    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    # Z(60000, 10) y(60000,) sum_exp_z(60000,)
    exp_z = ndl.exp(Z)
    sum_exp_z = ndl.summation(exp_z, axes=(-1,))
    log_sum_exp_z = ndl.log(sum_exp_z)
    b = Z.shape[0]
    z_mul = ndl.multiply(Z, y_one_hot)
    z_y  = ndl.summation(z_mul, axes=(-1,))
    loss = ndl.summation(log_sum_exp_z - z_y, axes=(-1,))
    loss = loss/b

    return loss
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).
    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch
    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    n = X.shape[0]
    m = W2.shape[1]
    y_one_hot = np.zeros((n,m))
    y_one_hot[np.arange(n), y] = 1
    step = n // batch
    for i in range(0, step, 1):
        start = i * batch
        end = min(start + batch, n)
        x = ndl.Tensor(X[start: end])
        y = ndl.Tensor(y_one_hot[start: end, :])
        Z1 = ndl.matmul(x, W1)
        Z1 = ndl.relu(Z1)
        Z2 = ndl.matmul(Z1, W2)
        # loss
        loss = softmax_loss(Z2, y)
        loss.backward()
        # grad
        W1 = ndl.Tensor(W1.numpy() - lr * W1.grad.numpy())
        W2 = ndl.Tensor(W2.numpy() - lr * W2.grad.numpy())
    return W1, W2
    ### END YOUR SOLUTION


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h,y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)