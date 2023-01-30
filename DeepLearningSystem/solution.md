# reshape

## reference
- https://blog.csdn.net/wuzhongqiang/article/details/107653655
- http://doraemonzzz.com/2022/10/17/2022-10-17-Deep-Learning-Systems-HW1/#Question-5-Softmax-loss
- https://github.com/YuanchengFang/dlsys_solution/blob/master/
- https://blog.csdn.net/weixin_43889476/article/details/123794879
  - sudo sh cuda_8.0.61_375.26_linux.run
  - ~/.bashrc
  - tar zxvf cudnn-8.0-linux-x64-v7.1.tgz
    - sudo cp cuda/include/cudnn.h /usr/local/cuda-8.0/include/
    - sudo cp cuda/lib64/libcudnn* /usr/local/cuda-8.0/lib64/
    - sudo chmod a+r /usr/local/cuda-8.0/include/cudnn.h 
  - nvcc -V


- <font color=red>No matter how the shape changes, the underlying buffer remains unchanged</font>
  - buffer is a one-dimensional array that never changes
  - The changing shape is communicated through the view
  - The axis whose value is only 0 is a free axis, which can transform any dimension

## example

- s0
  - a = np.array(0,24,2)

  | a | 0 | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 | 18 | 20 | 22 |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|
  | i | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |

  - a[6] = 12

- s1
  - a = a.reshape(2, 6)

  | a | 0 | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 | 18 | 20 | 22 |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|
  | i | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 1 | 1 |
  | j | 0 | 1 | 2 | 3 | 4 | 5 | 0 | 1 | 2 | 3 | 4 | 5 |

  - a[1,0] = 12

- s2
  - a = a.reshape(2, 3, 2)

  | a | 0 | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 | 18 | 20 | 22 |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|
  | i | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 1 | 1 |
  | j | 0 | 0 | 1 | 1 | 2 | 2 | 0 | 0 | 1 | 1 | 2 | 2 |
  | k | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 1 |

  - a[1,0,0] = 12


- s2
  - a = a.reshape(1, 2, 3, 2)

  | a | 0 | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 | 18 | 20 | 22 |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|
  | i | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
  | i | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 1 | 1 |
  | j | 0 | 0 | 1 | 1 | 2 | 2 | 0 | 0 | 1 | 1 | 2 | 2 |
  | k | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 1 |

  - a[0,1,0,0] = 12
  - Value in i are all 0, i is a free axis