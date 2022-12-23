# Basic Elementary Function Derivation
- (C)' = 0
- $ (x^a)' = a x^{a-1} $
- sin(x)' = cos(x)
- cos(x)' = -sin(x)
- $ tan(x)' = sec^2(x) $
- $ cot(x)' = -csc^2(x) $
- $ sec(x)' = sec(x)tan(x) $
- $ csc(x)' = -csc(x)cot(x) $
- $ (a^x)' = a^x ln(a) $
- $ (e^x)' = e^x $
- $ (log_a(x))' = \frac{1}{x ln(a)} $
- $ ln(x)' = \frac1x $
- $ (arcsin(x))' = \frac{1}{\sqrt{1-x^2}} $
- $ (arccos(x))' = -\frac{1}{\sqrt{1-x^2}} $
- $ (arctan(x))' = \frac{1}{1+x^2} $
- $ (arccot(x))' = -\frac{1}{1+x^2} $
- $ (arcsec(x))' = \frac{1}{ \vert x \vert \sqrt{x^2-1}} $
- $ (arccsc(x))' = -\frac{1}{ \vert x \vert \sqrt{x^2-1}} $

# function
- u = u(x)
- v = v(x)
- (u+v)' = u' + v'
- (uv)' = u'v + uv'
- (Cu)' = Cu'
  - C = constant
- $ (\frac{u}{v})' = \frac{u'v - uv'}{v^2} $

# log
- $ log_{(a^k)}(M^n) = \frac{n}{k}log_a(M) $
- $ log_a(N) = \frac{log_b(N)}{log_b(a)} = \frac{ln_e(N)}{ln_e(a)} = \frac{lg_{10}(N)}{lg_{10}((a)} $

---
---


# EWiseAdd
- compute
  - f = a + b
- gradient
  - $ a_g = \frac{\partial f}{\partial out} * \frac{\partial f}{\partial a} = o_g * 1 = o_g $

  - $ b_g = \frac{\partial f}{\partial out} * \frac{\partial f}{\partial b} = o_g * 1 = o_g $

# AddScalar
- compute
  - f = a + scalar
- gradient
  - $ a_g = \frac{\partial f}{\partial out} * \frac{\partial f}{\partial a} = o_g * 1 = o_g $

# EWiseMul
- compute
  - f = a * b
- gradient
  - $ a_g = \frac{\partial f}{\partial out} * \frac{\partial f}{\partial a} = o_g * b $

  - $ b_g = \frac{\partial f}{\partial out} * \frac{\partial f}{\partial b} = o_g * a $

# MulScalar
- compute
  - f = a * scalar
- gradient
  - $ a_g = \frac{\partial f}{\partial out} * \frac{\partial f}{\partial a} = o_g * scalar $

# PowerScalar
- compute
  - $ f = a ^ {scalar} $
- gradient
  - $ a_g = \frac{\partial f}{\partial out} * \frac{\partial f}{\partial a} = o_g * scalar * a ^ {scalar-1}  $

# EWiseDiv
- compute
  - $ f = \frac{a}{b} $
- gradient
  - $ a_g = \frac{\partial f}{\partial out} * \frac{\partial f}{\partial a} = o_g * \frac1b $

  - $ a_g = \frac{\partial f}{\partial out} * \frac{\partial f}{\partial b} = o_g * -1 * \frac{a}{b^2} = o_g * \frac{-a}{b^2} $

# DivScalar
- compute
  - $ f = \frac{a}{scalar} $
- gradient
  - $ a_g = \frac{\partial f}{\partial out} * \frac{\partial f}{\partial a} = o_g * \frac1{scalar} $

# Transpose
- compute
  - f = a.transpose(axes)
- gradient
  - a_g = o_g.transpose(argsort(axes))

# Reshape
- compute
  - f = a.reshape(shape)
- gradient
  - a_g = o_g.reshape(a.shape)

# BroadcastTo
- compute
  - f = broadcast_to(a, shape)
- gradient
  - a_g = sum(o_g)

# Summation
- compute
  - f = sum(a, axes)
- gradient
  - a_g = broadcast_to(o_g, a.shape)

# MatMul
- compute
  - f = matmul(a, b)
    - a[m,k]
    - b[k,n]
    - f[m,n]
- gradient
  - a_g = matmul(o_g, b.T)
    - o_g[m,n]
    - b.T[n,k]
    - a_g[m,k]
  - b_g = matmul(a.T, o_g)
    - a.T[k,m]
    - o_g[m,n]
    - b_g[k,n]

# Negate
- compute
  - f = negative(a)
- gradient
  - a_g = negative(o_g)

# Log
- compute
  - f = log(a)
- gradient
  - $ a_g = \frac{\partial f}{\partial out} * \frac{\partial f}{\partial a} = o_g*\frac{1}{a} $

# Exp
- compute
  - f = exp(a)
- gradient
  - $ a_g = \frac{\partial f}{\partial out} * \frac{\partial f}{\partial a} = o_g*exp(a) $

# ReLU
- compute
  - f = maximum(a, 0)
- gradient
  - $ a_g = \frac{\partial f}{\partial out} * \frac{\partial f}{\partial a} = o_g*([a<=0]=0) $

# LogSoftmax
- compute
  - f = log(softmax(a))
- gradient
  - $ a_g = \frac{\partial f}{\partial out} * \frac{\partial f}{\partial a} = o_g * \frac{1}{o} * (o-o^2) $

