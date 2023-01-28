import numpy as np
from .autograd import Tensor
import gzip
import struct
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing

        img[:, ::-1, :]: reverse H, horizontal flip
        img[::-1, :, :]: reverse W, vertical flip
        img[:, :, ::-1]: BGR -> RGB
        """ 
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        if flip_img:
            # return np.flip(img, axis=1)
            img = img[:, ::-1, :]
        return img
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)
        ### BEGIN YOUR SOLUTION
        result = np.zeros_like(img)
        H, W = img.shape[0], img.shape[1]
        # NOTE: when shift out of bounds, just return zeros 
        if abs(shift_x) >= H or abs(shift_y) >= W:
            return result
        st_1, ed_1 = max(0, -shift_x), min(H - shift_x, H)
        st_2, ed_2 = max(0, -shift_y), min(W - shift_y, W)
        img_st_1, img_ed_1 = max(0, shift_x), min(H + shift_x, H)
        img_st_2, img_ed_2 = max(0, shift_y), min(W + shift_y, W)
        result[st_1:ed_1, st_2:ed_2, :] = img[img_st_1:img_ed_1, img_st_2:img_ed_2, :]
        return result
        ### END YOUR SOLUTION


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    https://stackoverflow.com/questions/15474159/shuffle-vs-permute-numpy
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        # self.n = len(dataset)
        indices = np.arange(len(dataset))
        if not self.shuffle:
            self.ordering = np.array_split(np.arange(len(dataset)), 
                                           range(batch_size, len(dataset), batch_size))
        else:
            np.random.shuffle(indices)
            self.ordering = np.array_split(indices, 
                                           range(batch_size, len(dataset), batch_size))                                           

    def __iter__(self):
        ### BEGIN YOUR SOLUTION
        # self.index = 0
        # if self.shuffle:
        #     indexes = np.arange(self.n)
        #     np.random.shuffle(indexes)
        #     self.ordering = np.array_split(indexes, range(self.batch_size, self.n, self.batch_size))
        self.start = 0
        ### END YOUR SOLUTION
        return self

    def __next__(self):
        ### BEGIN YOUR SOLUTION
        # if self.index == len(self.ordering):
        #     raise StopIteration

        # res = [Tensor(x) for x in self.dataset[self.ordering[self.index]]]
        # self.index += 1
        
        # return tuple(res)
        if self.start == len(self.ordering):
            raise StopIteration
        a = self.start
        self.start += 1
        samples = [Tensor(x) for x in self.dataset[self.ordering[a]]]
        return tuple(samples)        
        ### END YOUR SOLUTION


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


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)
        self.X, self.y = parse_mnist(image_filename, label_filename)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        x = self.X[index]
        y = self.y[index]
        n = len(x.shape)
        if n == 1:
            x = x.reshape(28, 28, -1)
            x = self.apply_transforms(x)
            x = x.reshape(28, 28, 1)
        else:
            m = x.shape[0]
            x = x.reshape(m, 28, 28, -1)
            for i in range(m):
                x[i] = self.apply_transforms(x[i])

        return x, y
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.X.shape[0]
        ### END YOUR SOLUTION

class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])
