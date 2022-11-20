# reshape

## reference
- https://blog.csdn.net/wuzhongqiang/article/details/107653655

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