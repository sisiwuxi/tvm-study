import struct
import numpy as np
import gzip
import matplotlib.pyplot as plt
try:
    from simple_ml_ext import *
except:
    pass


def add(x, y):
    """ A trivial 'add' function you should implement to get used to the
    autograder and submission system.  The solution to this problem is in the
    the homework notebook.

    Args:
        x (Python number or numpy array)
        y (Python number or numpy array)

    Return:
        Sum of x + y
    """
    ### BEGIN YOUR CODE
    return x + y
    ### END YOUR CODE


def parse_mnist(image_filename, label_filename):
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
                maximum value of 1.0. The normalization should be applied uniformly
                across the whole dataset, _not_ individual images.

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    
    # with open(image_filename,'rb') as f:
    #     magic, size = struct.unpack(">II", f.read(8))
    #     nrows, ncols = struct.unpack(">II", f.read(8))
    #     data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    #     data = data.reshape((size, nrows, ncols))
    #     plt.imshow(image[0,:,:], cmap='gray')
    #     plt.savefig('./mnist_0.png')    
    with gzip.open(image_filename, 'rb') as f:
        # (2051, 60000), 0x00000803
        magic, size = struct.unpack(">II", f.read(8))
        # (28, 28)
        nrows, ncols = struct.unpack(">II", f.read(8))
        # (47040000,)
        image = np.frombuffer(f.read(), dtype=np.dtype(np.uint8).newbyteorder('>'))
        # (60000, 784)
        image = image.reshape((size, nrows*ncols))
    # In normalization, we map the minimum feature value to 0 and the maximum to 1. 
    # Hence, the feature values are mapped into the [0, 1] range:
    min = image.min()
    max = image.max()
    normalize_image = (np.float32)(image - min)/(max - min)
    # # In standardization, we don’t enforce the data into a definite range. 
    # # Instead, we transform to have a mean of 0 and a standard deviation of 1:
    # # It not only helps with scaling but also centralizes the data.
    # # In general, standardization is more suitable than normalization in most cases.
    # mean = image.mean()
    # std = image.std()
    # standardization_image = (np.float32)(image - mean) / std

    with gzip.open(label_filename,'rb') as f:
        # (2049, 60000), 0x00000801
        magic, size = struct.unpack(">II", f.read(8))
        # (60000,)
        label = np.frombuffer(f.read(), dtype=np.dtype(np.uint8).newbyteorder('>'))
        # (60000,)
        label = label.reshape((size,)) # (Optional)
        # array([5, 0, 4, ..., 5, 6, 8], dtype=uint8)
        # print(label)
    return normalize_image, label
    ### END YOUR CODE

def softmax_loss(Z, y):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (np.ndarray[np.float32]): 2D numpy array of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (np.ndarray[np.int8]): 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """
    ### BEGIN YOUR CODE
    # Z(60000, 10) y(60000,) sum_exp_z(60000,)
    sum_exp_z = np.sum(np.exp(Z), axis=-1)
    log_sum_exp_z = np.log(sum_exp_z)
    b = Z.shape[0]
    # z_y(60000,)
    z_y = Z[np.arange(b), y]
    # np.log(sum_exp_z)(60000,)
    loss = np.mean(log_sum_exp_z - z_y)

    return loss
    ### END YOUR CODE

def net_input(X, W, b):
    return (X.dot(W) + b)    
def softmax(z):
    # return (np.exp(z.T) / np.sum(np.exp(z), axis=1)).T
    return np.exp(z) / np.sum(np.exp(z), axis=-1, keepdims=True)
def to_classlabel(z):
    return z.argmax(axis=1)
def cross_entropy(output, y_target):
    return - np.sum(np.log(output) * (y_target), axis=1)
def cost(output, y_target):
    return np.mean(cross_entropy(output, y_target))

def softmax_regression_epoch(X, y, theta, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for softmax regression on the data, using
    the step size lr and specified batch size.  This function should modify the
    theta matrix in place, and you should iterate through batches in X _without_
    randomizing the order.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        theta (np.ndarrray[np.float32]): 2D array of softmax regression
            parameters, of shape (input_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    # X(60000, 784) y(60000,) theta(784, 10)
    # Iy
    Y = (np.arange(np.max(y) + 1) == y[:, None]).astype(float)
    n = X.shape[0]
    step = n // batch
    for i in range(step + 1):
        start = i * batch
        end = min(start + batch, n)
        if start == end:
            break
        x = X[start: end]
        y = Y[start: end, :]
        z = softmax(x@theta)
        # loss = z - Iy
        G = z - y
        # grad = x.T @ (z - Iy)
        theta_grad = (1/batch) * (x.T@G)
        theta -= lr * theta_grad
    return

def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).  It should modify the
    W1 and W2 matrices in place.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (np.ndarray[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (np.ndarray[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    # X(60000, 784) y(60000,) W1(784, 100) W2(100, 10)
    # Iy
    Y = (np.arange(np.max(y) + 1) == y[:, None]).astype(float)
    n = X.shape[0]
    step = n // batch
    for i in range(0, step, 1):
        start = i * batch
        end = min(start + batch, n)
        x = X[start: end]
        y = Y[start: end, :]
        Z1 = x@W1
        Z1 = np.maximum(np.zeros_like(Z1), Z1)
        Z2 = Z1@W2
        # loss = z - Iy
        G2 = softmax(Z2) - y
        G1 = G2@W2.T
        G1[Z1 <= 0] = 0
        # grad = x.T @ (z - Iy)
        W1_grad = (1/batch) * (x.T@G1)
        W2_grad = (1/batch) * (Z1.T@G2)
        W1 -= lr * W1_grad
        W2 -= lr * W2_grad
    ### END YOUR CODE
    return


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper funciton to compute both loss and error"""
    return softmax_loss(h,y), np.mean(h.argmax(axis=1) != y)


def train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.5, batch=100, cpp=False):
    """ Example function to fully train a softmax regression classifier """
    # (784, 10)
    theta = np.zeros((X_tr.shape[1], y_tr.max()+1), dtype=np.float32)
    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        if not cpp:
            softmax_regression_epoch(X_tr, y_tr, theta, lr=lr, batch=batch)
        else:
            softmax_regression_epoch_cpp(X_tr, y_tr, theta, lr=lr, batch=batch)
        train_loss, train_err = loss_err(X_tr @ theta, y_tr)
        test_loss, test_err = loss_err(X_te @ theta, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |".format(epoch, train_loss, train_err, test_loss, test_err))


def train_nn(X_tr, y_tr, X_te, y_te, hidden_dim = 500,
             epochs=10, lr=0.5, batch=100):
    """ Example function to train two layer neural network """
    n, k = X_tr.shape[1], y_tr.max() + 1
    np.random.seed(0)
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)

    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        nn_epoch(X_tr, y_tr, W1, W2, lr=lr, batch=batch)
        train_loss, train_err = loss_err(np.maximum(X_tr@W1,0)@W2, y_tr)
        test_loss, test_err = loss_err(np.maximum(X_te@W1,0)@W2, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))



if __name__ == "__main__":
    # (60000, 784), (60000,)
    X_tr, y_tr = parse_mnist("../data/train-images-idx3-ubyte.gz",
                             "../data/train-labels-idx1-ubyte.gz")
    # (10000, 784), (10000,)
    X_te, y_te = parse_mnist("../data/t10k-images-idx3-ubyte.gz",
                             "../data/t10k-labels-idx1-ubyte.gz")
    print("Training softmax regression")
    train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.1)
    print("Training softmax regression through call library")
    train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.1, cpp=True)

    print("\nTraining two layer neural network w/ 100 hidden units")
    train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=100, epochs=20, lr=0.2)
