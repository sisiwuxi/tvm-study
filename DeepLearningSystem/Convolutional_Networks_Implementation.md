# Implementing convolutions
- This notebook will walk you through the process of implementing a reasonably efficient convolutional. We'll do this using native numpy for the purposes of illustration, but in the homework, you'll need to implement these all using your own straight-C (and GPU) implementations.

# Convolutions
- Here we will build up some of the basic approaches for convolution, from a simple all-for-loop algorithm to an algorithm that uses a single matrix multiplication plus resize operations.

# Storage order
- In the simple fully-connected networks we have been developing so far, hidden units are typically simply represented as vectors, i..e., a quantity $ z \in R^n $, or when representing an entire minibatch, a matrix $ z \in R^{B,n} $. But when we move to convolutional networks, we need to include additional structure in the hidden unit. This is typically done by representing each hidden vector as a 3D array, with dimensions height x width x channels, or in the minibatch case, with an additional batch dimension. That is, we could represent a hidden unit as an array.
  - float Z[BATCHES][HEIGHT][WIDTH][CHANNELS]
- The format above is referred to as NHWC format (number(batch)-height-width-channel). However, there are other ways we can represent the hidden unit as well. For example, PyTorch defaults to the NCHW format (indexing over channels in the second dimension, then height and width), though it can also support NHWC in later versions. There are subtle but substantial differences in the performance for each different setting: convolutions are typically faster in NHWC format, owing to their ability to better exploit tensor cores; but NCHW format is typically faster for BatchNorm operation (because batch norm for convolutional networks operates over all pixels in an individual channel).
- Although less commonly discussed, there is a simliar trade-off to be had when it comes to storing the convolutional weights (filter) as well. Convolutional filters are specified by their kernel size (which can technically be different over different height and width dimensions, but this is quite uncommon), their input channels, and their output channels. We'll store these weights in the form:
  - float weights[KERNEL_SIZE][KERNEL_SIZE][IN_CHANNELS][OUT_CHANNELS];
- Again, PyTorch does things a bit differently here (for no good reason, as far as I can tell, it was just done that way historically), storing weight in the order OUT_CHANNELS x IN_CHANNELS x KERNELS_SIZE x KERNEL_SIZE.
# Convolutions with simple loops
- Let's begin by implementing a simple convolutional operator. We're going to implement a simple version, which allows for different kernel sizes but which doesn't have any built-in padding: to implement padding, you'd just explicitly form a new ndarray with the padding built in. This means that if we have an HxW input image and convolution with kernel size K, we'll end up with a (H-K+1)x(W-K+1) image.
- Although it's "cheating" in some sense, we're going to use PyTorch as a reference implementation of convolution that we will check against. However, since PyTorch, as mentioned above, uses the NCHW format (and stores the convolutional weights in a different ordering as well), and we'll use the NHWC format and the weights ordering stated above, we will need to swap things around for our reference implementation.
- Now let's consider the simplest possible implementation of a convolution, that just does the entire operation using for loops.
- We can check to make sure this implementation works by comparing to the PyTorch reference implementation.
- The implementation works, but (not surprisingly, since you would never want to actually have a 7-fold loop in interpreted code), the PyTorch version is much much faster; your mileage may vary (YMMV), but on my laptop, the naive implementation is more than 2000 times slower.
  ```python
  def conv_naive(F, W):
    N,Hi,Wi,Ci = F.shape
    R,S,Ci,Co = W.shape
    out = np.zeros((N,Hi-R+1,Wi-S+1,Co))
    for n in range(N):
      for ci in range(Ci):
        for co in range(Co):
          for h in range(Hi-R+1):
            for w in range(Wi-S+1):
              for r in range(R):
                for s in range(S):
                  out[n,h,w,co] += F[n,h+r,w+s,ci] * W[r,s,ci,co]
  return out
  ```

# Convolutions as matrix mulitplications
- Ok, but, no one is going to actually implement convolutions elementwise in Python. Let's see how we can start to do much better. The simplest way to make this much faster (and frankly, a very reasonable implementation of convolution) is to perform it as a sequence of matrix multiplications. Remember that a kernel size K=1 convolution is equivalent to performing matrix multiplication over the channel dimensions. That is, suppose we have the following convolution.
- Then we could implement the convolution using a single matrix multiplication.
- We're here exploiting the nicety that in numpy, when you compute a matrix multiplication by a multi-dimensional array, it will treat the leading dimensions all as rows of a matrix. That is, the above operation would be equivalent to:
- This strategy immediately motivates a very natural approach to convolution: we can iterate over just the kernel dimensions R and S, and use matrix multiplication to perform the convolution.
  ```python
  for kh in range(Kh):
    for kw in range(Kw):
      out += F[:,kh:kh+Hi-Kh+1,kw:kw+Wi-Kw+1,:] @ W[kh,kw]
  ```
- This works as well, as (as expected) is much faster, starting to be competetive even with the PyTorch version (about 2-3x slower on my machine). Let's in fact increase the batch size a bit to make this a more lengthy operation.

# Manipulating matrices via strides
- Before implementing convolutions via im2col, let's consider an example that actually has nothing to do with convolution. Instead, let's consider the efficient matrix multiplication operations that we discussed in an earlier lecture. Normally we think of storing a matrix as a 2D array:
  - float A[M][N]
- In the typical row-major format, this will store each N dimensional row of the matrix one after the other in memory. However, recall that in order to make better use of the caches and vector operations in modern CPUs, it was beneficial to lay out our matrix memory groups by individual small "tiles", so that the CPU vector operations could efficiently access operators:
  - float A[M/TILE][N/TILE][TILE][TILE]
- where TILE is some small constant (like 4), which allows the CPU to use its vector processor to perform very efficient operations on TILE x TILE blocks. Importantly, what enables this to be so efficient is that in the standard memory ordering for an ND array, this grouping would locate all TILE x TILE block consecutively in memroy, so they could quickly be loaded in and out of cache / registers / etc.
- How exactly would we convert a matrix to this form? You could imagine how to manually copy from one matrix type to another, but it would be rather cumbersome to write this code each time you wanted to experiment with different (and in order for the code to be efficient, you'd need to write it in C/C++ as well, which could get to be a pain). Instead, we're going to show you how to do this using the handy function np.lib.stride_tricks.as_strided(), which lets you create new matrices by manually manipulating the strides of a matrix but not changing the data; we can then use np.ascontiguousarray() to lay out the memory sequentially. This sets of tricks let us rearrange matrices fairly efficiently in just one or two lines of numpy code.

# An example: a 6x6 2D array
- To see how this works, let's consider an example 6x6 numpy array.
  ```python
    n = 6
    A = np.arange(n**2, dtype=np.float32).reshape(n,n)
    print(A)
    [[ 0.  1.  2.  3.  4.  5.]
    [ 6.  7.  8.  9. 10. 11.]
    [12. 13. 14. 15. 16. 17.]
    [18. 19. 20. 21. 22. 23.]
    [24. 25. 26. 27. 28. 29.]
    [30. 31. 32. 33. 34. 35.]]    
  ```
- This array is layed out in memory by row. It's actually a bit of a pain to access the underlying raw memory of a numpy array in Python (numpy goes to great things to try to prevent you from doing this, but we can see how the array is layed out using the following code):
  ```python
    print(np.frombuffer(ctypes.string_at(A.ctypes.data, A.nbytes), dtype=A.dtype, count=A.size))
    [ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11. 12. 13. 14. 15. 16. 17.
    18. 19. 20. 21. 22. 23. 24. 25. 26. 27. 28. 29. 30. 31. 32. 33. 34. 35.]
  ```

- A few lectures ago, we discussed the use of the strides structure as a way to lay out n-dimensional arrays in memory. In order to access the A[i][j] element of a 2D array, for instance, we would access the memory location at:
  - A.bytes[i * strides[0] + j * strides[1]]
- The same can be done e.g., with a 3D tensor, accessing A[i][j][k] at memory location:
  - A.bytes[i * strides[0] + j * strides[1] + k * strides[2]]

- For an array in row-major format, we would thus have
  ```python
    strides[0] = num_cols
    strides[1] = 1
  ```
- We can look at the strides of the array we have created using the .strides property.
  ```python
    print(A.strides)
    (24, 4)
  ```
  - A[6,6].strides = (6*4, 1*4)
- Note that numpy, somewhat unconventionally, actually uses strides equal to the total number of bytes, so these numbers are all multiplied by 4 from the above, because a float32 type takes up 4 bytes.

# Tiling a matrix using strides
- Now let's consider how to create a tiled form of the A array by just changing the strides. For simplicity, let's assume we want to tile into 2x2 blocks, and thus we want to convert A into a 3 x 3 x 2 x 2 array. What would the strides be in this case? In other words, if we accessed the element A[i][j][k][l], how would this index into a memory location in the array as layed out above? Incrementing the first index, i, would move down two rows in the matrix, so strides[0] = 12; similarly, incrementing the second index j would move over two columns, so strides[1]=2. Things get a bit tricker next, but are still fairly straightforward: incrementing the next index k moves down one row in the matrix, so strides[2]=6, and finally incrementing the last index l just moves us over one column, so strides[3]=1.
  - A[6,6] -> A[3,3,2,2]
    ```python
      [
      [ 0.  1.][ 2.  3.][ 4.  5.]
      [ 6.  7.][ 8.  9.][ 10. 11.]

      [12. 13.][ 14. 15.][ 16. 17.]
      [18. 19.][ 20. 21.][ 22. 23.]

      [24. 25.][ 26. 27.][ 28. 29.]
      [30. 31.][ 32. 33.][ 34. 35.]
      ]
    ```
  - A[6,6]
  - A.bytes[i*strides[0] + j*strides[1] + k*strides[2] + l*strides[3]]
  - strides[0] = 3*2*2 = 12
  - strides[1] = 2
  - strides[2] = 3*2 = 6
  - strides[3] = 1

Let's create a matrix with this form using the np.lib.stride_tricks.as_strided(). This function lets you specify the shape and stride of a new matrix, created from the same memory as an old matrix. That is, it doesn't do any memory copies, so it's very efficient. But you also have to be careful when you use it, because it's directly creating a new view of an existing array, and without proper care you could e.g., go outside the bounds of the array.

- Here's how we can use it to create the tiled view of the matrix A.
    ```python
    B = np.lib.stride_tricks.as_strided(A, shape=(3,3,2,2), strides=np.array((12,2,6,1))*4)
    print(B)  
    B[3,3,2,2]  
    [[[[ 0.  1.]
      [ 6.  7.]]

      [[ 2.  3.]
      [ 8.  9.]]

      [[ 4.  5.]
      [10. 11.]]]


    [[[12. 13.]
      [18. 19.]]

      [[14. 15.]
      [20. 21.]]

      [[16. 17.]
      [22. 23.]]]


    [[[24. 25.]
      [30. 31.]]

      [[26. 27.]
      [32. 33.]]

      [[28. 29.]
      [34. 35.]]]] 
    print(B.strides)
    (48, 8, 24, 4)
    ```
- Parsing numpy output for ND array isn't the most intuitive thing, but if you look you can see that these basically lay out each 2x2 block of the matrix, as desired. However, can also see the fact that this call didn't change the actual memory layout by again inspecting the raw memory.
- In order to change reorder the memory so that the underlying matrix is continguous/compact (which is what we need for making the matrix multiplication efficient), we can use the np.ascontinugousarray() function.
  ```
    C = np.ascontiguousarray(B)
    print(C)
    print(np.frombuffer(ctypes.string_at(C.ctypes.data, size=C.nbytes), C.dtype, C.size))
    print(C.strides)

    [[[[ 0.  1.]
      [ 6.  7.]]

      [[ 2.  3.]
      [ 8.  9.]]

      [[ 4.  5.]
      [10. 11.]]]


    [[[12. 13.]
      [18. 19.]]

      [[14. 15.]
      [20. 21.]]

      [[16. 17.]
      [22. 23.]]]


    [[[24. 25.]
      [30. 31.]]

      [[26. 27.]
      [32. 33.]]

      [[28. 29.]
      [34. 35.]]]]
    [ 0.  1.  6.  7.  2.  3.  8.  9.  4.  5. 10. 11. 12. 13. 18. 19. 14. 15.
    20. 21. 16. 17. 22. 23. 24. 25. 30. 31. 26. 27. 32. 33. 28. 29. 34. 35.]
    (48, 16, 8, 4)
  ```
  C[3,3,2,2]
  - strides[0] = 3*2*2 = 12
  - strides[1] = 2*2 = 4
  - strides[2] = 2 = 2
  - strides[3] = 1  

# Convolutions via im2col
- Let's consider finally the "real" way to implement convolutions, which will end up being about as fast as PyTorch's implementation. Essentially, we want to bundle all the computation needed for convolution into a single matrix multiplication, which will then leverage all the optimizations that we can implement for normal matrix multiplication.
- They key approach to doing this is called the im2col operator, which "unfolds" a 4D array into exactly the form needed to perform multiplication via convolution. Let's see an example of how this works using a simple 2D array, before we move to the 4D case. Let's consider the following array we used above in the first section.
- And let's consider convolting with a 3x3 filter.
- Recall that a convolution will multiply this filter with every 3x3 block in the image. So how can we extract every such 3x3 block. The key will be to form a (Hi-Kh+1)x(Wi-Kw+1)xKhxKw array, that contains all of these blocks, then flatten it to a matrix we can multiply by the filter (this is the same process we did mathematically in the previous lecture on convolutions for 1D convolutions, but we're now doing to do it for real for the 2D case). But how can we go about creating this array of all blocks, short of manual copying. Fortunately, it turns out that the as_strided() call we talked about above is actually exactly what we needed for this.
- Specifically, if we created a new view in the matrix, of size (4,4,3,3), how can we use as_strided() to return the matrix we want? Well, note that the first two dimenions will have strides of 6 and 1, just like in the regular array: incrementing the first index by 1 will move to the next row, and incrementing the next will move to the next column. But interestingly (and this is the "trick"), the third and fourth dimensions also have strides of 6 and 1 respectively, because incrementing the third index by one also moves to the next row, and similarly for the fourth index. Let's see what this looks like in practice.
  ```
    B = np.lib.stride_tricks.as_strided(A, shape=(4,4,3,3), strides=4*(np.array((6,1,6,1))))
    print(B)
    [[[[ 0.  1.  2.]
      [ 6.  7.  8.]
      [12. 13. 14.]]

      [[ 1.  2.  3.]
      [ 7.  8.  9.]
      [13. 14. 15.]]

      [[ 2.  3.  4.]
      [ 8.  9. 10.]
      [14. 15. 16.]]

      [[ 3.  4.  5.]
      [ 9. 10. 11.]
      [15. 16. 17.]]]


    [[[ 6.  7.  8.]
      [12. 13. 14.]
      [18. 19. 20.]]

      [[ 7.  8.  9.]
      [13. 14. 15.]
      [19. 20. 21.]]

      [[ 8.  9. 10.]
      [14. 15. 16.]
      [20. 21. 22.]]

      [[ 9. 10. 11.]
      [15. 16. 17.]
      [21. 22. 23.]]]


    [[[12. 13. 14.]
      [18. 19. 20.]
      [24. 25. 26.]]

      [[13. 14. 15.]
      [19. 20. 21.]
      [25. 26. 27.]]

      [[14. 15. 16.]
      [20. 21. 22.]
      [26. 27. 28.]]

      [[15. 16. 17.]
      [21. 22. 23.]
      [27. 28. 29.]]]


    [[[18. 19. 20.]
      [24. 25. 26.]
      [30. 31. 32.]]

      [[19. 20. 21.]
      [25. 26. 27.]
      [31. 32. 33.]]

      [[20. 21. 22.]
      [26. 27. 28.]
      [32. 33. 34.]]

      [[21. 22. 23.]
      [27. 28. 29.]
      [33. 34. 35.]]]]
  ```

- This is exactly the 4D array we want. Now, if we want to compute the convolution as a "single" matrix multiply, we just flatten reshape this array to a (4.4)x(3.3) matrix, reshape the weights to a 9 dimensional vector (the weights will become a matrix again for the case of multi-channel convolutions), and perform the matrix multiplication. We then reshape the resulting vector back into a 4x4 array to perform the convolution.
  ```python
    A = np.arange(n**2, dtype=np.float32).reshape(n,n)
    B = np.lib.stride_tricks.as_strided(A, shape=(4,4,3,3), strides=4*(np.array((6,1,6,1))))
    C = B.reshape(16,9)
  
    W = np.arange(9, dtype=np.float32).reshape(3,3)
    out = (C @ W.reshape(9)).reshape(4,4)
  ```
# A critical note on memory efficiency
- There is a very crucial point to make regarding memory efficiency of this operation. While reshaping W into an array (or what will be a matrix for multi-channel convolutions) is "free", in that it doesn't allocate any new memory, reshaping the B matrix above is very much not a free operation. Specifically, while the strided form of B uses the same memory as A, once we actually convert B into a 2D matrix, there is no way to represent this data using any kind of strides, and we have to just allocate the entire matrix. This means we actually need to form the full im2col matrix, which requires  more memory than the original image, which can be quite costly for large kernel sizes.
- For this reason, in practice it's often the case that the best modern implementations won't actually instatiate the full im2col matrix, and will instead perform a kind of "lazy" formation, or specialize the matrix operation natively to im2col matrices in their native strided form. These are all fairly advanced topics that we won't deal with any further in the course, because for our purposes, it will be sufficient to just allocate this matrix and then quickly deallocate it after we perform the convolution (remember that we aren't e.g., doing backprop through the im2col operation).

# im2col for multi-channel convolutions
- So how do we actually implement an im2col operation for real multi-channel, minibatched convolutions? It turns out the process is not much more complicated. Instead of forming a 4D (Hi-Kh+1)x(Wi-Kw+1)*Kh*Kw array, we form a 6D Nx(Hi-Kh+1)x(Wi-Kw+1)xKhxKwxCi array (leaving the minibatch and channel dimensions untouched). And, after thinking about it for a bit, it should be pretty clear that we can apply the same trick by just repeating the strides for dimensions 1 and 2 (the height and width) for dimensions 3 and 4 (the KhxKw blocks), and leave the stirdes for the minibatch and channels unchanged. Furthermore, you don't even need to worry about manually computing the strides manually: you can just use the strides of the F input and repeat whatever they are.

- To compute the convolution, you then flatten the im2col matrix to a (N.(Hi-Kh+1).(Wi-Kw+1))x(Kh.Kw.Ci) matrix (remember, this operation is highly memory inefficient), flatten the weights array to a (Kh.Kw.Ci)xCo matrix, perform the multiplication, and resize back to the desired size of the final 4D array output. Here's the complete operation.
- Again, we can check that this version produces the same output as the PyTorch reference (or our other implementations, at this point):
- However, at this point we're finally starting to get competetive with PyTorch, taking only 25% more time than the PyTorch implementation on my machine.

# Final notes
- Hopefully this quick intro gave you a bit of appreciation and understanding of what is going on "under the hood" of modern convolution implementations. It hopefully also gave you some understanding how just how powerful stride manipulation can be, able to accomplish some very complex operations without actually needing to explicitly loop over matrices (though, as we'll see on the homework, a bit of the complexity is still being outsourced to the .reshape and it's implicit np.ascontinugousarray() call, which is not completely trivial; but we'll deal with this on the homework.