# reference
- Automatic Differentiation in Machine Learning: a Survey
  - https://www.jmlr.org/papers/volume18/17-468/17-468.pdf
- Tangent: Automatic Differentiation Using Source Code Transformation in Python
  - https://arxiv.org/pdf/1711.02712.pdf
- Differentiable Programming Tensor Networks
  - https://journals.aps.org/prx/pdf/10.1103/PhysRevX.9.031041
- A Differentiable Programming System to Bridge Machine Learning and Scientific Computing
  - https://arxiv.org/pdf/1907.07587.pdf

---
---

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

# op
## EWiseAdd
- compute
  - $ f(x,y) = x + y $
- gradient
  - $ \frac{\partial f(x,y)}{\partial x} = 1 $

  - $ \frac{\partial L}{\partial x} = \frac{\partial L}{\partial f(x,y)} \frac{\partial f(x,y)}{\partial x} = \frac{\partial L}{\partial f(x,y)} = o_g $

  - $ \frac{\partial L}{\partial y} = \frac{\partial L}{\partial f(x,y)} \frac{\partial f(x,y)}{\partial y} = \frac{\partial L}{\partial f(x,y)} = o_g $

## AddScalar
- compute
  - f(x, scalar) = x + scalar
- gradient
  - $ \frac{\partial f(x,scalar)}{\partial x} = 1 $

  - $ \frac{\partial L}{\partial x} = \frac{\partial L}{\partial f(x,scalar)} \frac{\partial f(x,scalar)}{\partial x} = \frac{\partial L}{\partial f(x,scalar)} * 1 = o_g $

## EWiseMul
- compute
  - $ f(x,y) = x \circ y $
- gradient
  - $ \frac{\partial f(x,y)}{\partial x} = y $

  - $ \frac{\partial L}{\partial x} = \frac{\partial L}{\partial f(x,y)} \frac{\partial f(x,y)}{\partial x} = \frac{\partial L}{\partial f(x,y)} y = o_g y $

  - $ \frac{\partial f(x,y)}{\partial y} = x $

  - $ \frac{\partial L}{\partial y} = \frac{\partial L}{\partial f(x,y)} \frac{\partial f(x,y)}{\partial y} = \frac{\partial L}{\partial f(x,y)} x = o_g x $

## MulScalar
- compute
  - f(x, scalar) = x * scalar
- gradient
  - $ \frac{\partial f(x,scalar)}{\partial x} = scalar $
 
  - $ \frac{\partial L}{\partial x} = \frac{\partial L}{\partial f(x,scalar)} \frac{\partial f(x,scalar)}{\partial x} = \frac{\partial L}{\partial f(x,scalar)} scalar = o_g scalar $

## PowerScalar
- compute
  - $ f(x, scalar) = x ^ {scalar} $
- gradient
  - $ \frac{\partial f(x,scalar)}{\partial x} = scalar * x ^{scalar-1} $

  - $ \frac{\partial L}{\partial x} = \frac{\partial L}{\partial f(x,scalar)} \frac{\partial f(x,scalar)}{\partial x} = \frac{\partial L}{\partial f(x,y)} scalar = o_g * scalar * x ^{scalar-1} $

## EWiseDiv
- compute
  - $ f(x,y) = \frac{x}{y} $
- gradient
  - $ \frac{\partial f(x,y)}{\partial x} = \frac1y $

  - $ \frac{\partial L}{\partial x} = \frac{\partial L}{\partial f(x,y)} \frac{\partial f(x,y)}{\partial x} = \frac{\partial L}{\partial f(x,y)} \frac1y = o_g \frac1y $

  - $ \frac{\partial f(x,y)}{\partial y} = x * \frac{-1}{y^2} $

  - $ \frac{\partial L}{\partial y} = \frac{\partial L}{\partial f(x,y)} \frac{\partial f(x,y)}{\partial y} = \frac{\partial L}{\partial f(x,y)} * x * \frac{-1}{y^2}  = o_g * \frac{-x}{y^2} $

## DivScalar
- compute
  - $ f(x, scalar) = \frac{x}{scalar} $
- gradient
  - $ \frac{\partial f(x,scalar)}{\partial x} = \frac1{scalar} $

  - $ \frac{\partial L}{\partial x} = \frac{\partial L}{\partial f(x,scalar)} \frac{\partial f(x,scalar)}{\partial x} = \frac{\partial L}{\partial f(x,y)} \frac1{scalar} = o_g * \frac1{scalar} $

## Transpose
- compute
  - f(x, axes) = x.transpose(axes)
- gradient
  - x_g = o_g.transpose(argsort(axes))

## Reshape
- compute
  - f(x, shape) = x.reshape(shape)
- gradient
  - x_g = o_g.reshape(x.shape)

## BroadcastTo
- compute
  - f(x, shape) = broadcast_to(a, shape)
- gradient
  - a_g = sum(o_g)

## Summation
- compute
  - f(x, axes) = sum(a, axes)
- gradient
  - a_g = broadcast_to(o_g, a.shape)

## MatMul
- compute
  - f(a,b) = matmul(a, b)
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

## Negate
- compute
  - f(x) = negative(x)
- gradient
  - $ \frac{\partial f(x)}{\partial x} = -1 $
 
  - $ \frac{\partial L}{\partial x} = \frac{\partial L}{\partial f(x)} \frac{\partial f(x)}{\partial x} = \frac{\partial L}{\partial f(x)} * -1 = o_g * -1 $

## Log
- compute
  - f(x) = log(x)
- gradient
  - $ \frac{\partial f(x)}{\partial x} = \frac{1}{x} $

  - $ \frac{\partial L}{\partial x} = \frac{\partial L}{\partial f(x)} \frac{\partial f(x)}{\partial x} = \frac{\partial L}{\partial f(x)} * \frac{1}{x} = o_g * \frac{1}{x} $

## Exp
- compute
  - f(x) = exp(x)
- gradient
  - $ \frac{\partial f(x)}{\partial x} = exp(x) $

  - $ \frac{\partial L}{\partial x} = \frac{\partial L}{\partial f(x)} \frac{\partial f(x)}{\partial x} = \frac{\partial L}{\partial f(x)} * exp(x) = o_g * exp(x) $

## ReLU
- compute
  - f(x) = maximum(x, 0)
- gradient
  - $ \frac{\partial f(x)}{\partial x} 
    = ([a<=0]=0) 
    = \begin{cases}
        0, & x \le 0 \\
        1, & x \gt 0
      \end{cases}
    $

  - $ \frac{\partial L}{\partial x} 
  = \frac{\partial L}{\partial f(x)} * \frac{\partial f(x)}{\partial x} 
  = \frac{\partial L}{\partial f(x)} *
    \begin{cases}
      0, & x \le 0 \\
      1, & x \gt 0
    \end{cases}
  $

## LogSoftmax
- compute
  - f(x) = log(softmax(x)) = $ LogSoftmax(x_i) = log(\frac{e^{x_i}}{\sum_j^K e^{x_j}}) $
  - $ log(softmax(x_i)) = log(\frac{C e^{x_i}}{C \sum_j^K e^{x_j}})
                    = log(\frac{e^{x_i + log(C)}}{\sum_j^K e^{x_j + log(C)}})
                    = log(\frac{e^{x_i - m}}{\sum_j^K e^{x_j - m}})
    $
    - $ m = -log(C) = max(x_i) $

- gradient
  - sum
    - $ sum(x) = \sum_{j=1}^K e^{x_j} = e^{x_1} + \sum_{j=2}^K e^{x_j} $
    - $ \frac{\partial sum(x)}{\partial x_1} = e^{x_1} $

    - $ sum(x) = \sum_{j=1}^K e^{x_j} = e^{x_1} + e^{x_2} + \sum_{j=3}^K e^{x_j} $
    - $ \frac{\partial sum(x)}{\partial x_2} = e^{x_2} $

    - ...
  - softmax
    - $ softmax(x_1) = \frac{e^{x_1}}{\sum_{j=1}^K e^{x_j}} = \frac{e^{x_1}}{sum(x)} $

    - $ \frac{\partial softmax(x_1)}{\partial x_1} = \frac{e^{x_1}sum(x) - e^{x_1}e^{x_1}}{sum^2(x)} 
      = \frac{e^{x_1}}{sum(x)} - {(\frac{e^{x_1}}{sum(x)})}^2 = softmax(x_1)-softmax^2(x_1) $

    - $ \frac{\partial softmax(x_1)}{\partial x_2} = \frac{0*sum(x) - e^{x_1}e^{x_2}}{sum^2(x)} = -\frac{e^{x_1}}{sum(x)} \frac{e^{x_2}}{sum(x)}
     = -softmax(x_1)*softmax(x_2) $

    - $ \frac{\partial softmax(x_i)}{\partial x_j} =
      \begin{cases}
        softmax(x_i)-softmax(x_i)^2, & i = j \\
        -softmax(x_i)softmax(x_j), & i \ne j
      \end{cases} 
      $

  - $ \frac{\partial log(x)}{\partial x} = \frac{1}{x} $

  - $ \frac{\partial L}{\partial x} 
  = \frac{\partial L}{\partial log(softmax(x))} * \frac{\partial log(softmax(x))}{\partial softmax(x)} * \frac{\partial softmax(x)}{\partial x} 
  = O_g * \frac{1}{softmax} * \begin{cases}
    softmax(x_i)-softmax(x_i)^2, & i = j \\
    -softmax(x_i)softmax(x_j), & i \ne j
    \end{cases}
  $
---
---

# backward computation

## auto-differentiation

- The general goal of reverse mode auto-differentiation is to compute the gradient of some downstream function $ l $ of $ f(x,y) $  with respect to x (or y). Written formally, we could write this as trying to compute
  - $ \frac{\partial L}{\partial x} = \frac{\partial L}{\partial f(x,y)} \frac{\partial f(x,y)}{\partial x} $

- The "incoming backward gradient" is precisely the term $ \frac{\partial L}{\partial f(x,y)} $, so we want our gradient() function to ultimately compute the product between this backward gradient the function's own derivative $ \frac{\partial f(x,y)}{\partial x} $
 
## example
- f(x1, x2) = ln(x1) + x1x2 - sin(x2)
- DAG(Directed Acyclic Graph)
```
           ln      +
x1 -> v_-1 --> v1 ---> v4
        \    //\        \
        \\|  /          \\/
        * v2           - v5 ---> f(x1,x2)
        //\             //\
        /      sin      /
x2 -> v0 ------------> v3

```

- forward trace
  - v-1 = x1 = 2
  - v0 = x2 = 5
  - v1 = ln(v-1) = ln2 = 0.693
  - v2 = v-1 v0 = 2*5 = 10
  - v3 = sin(v0) = sin(5) = 0.959
  - v4 = v1 + v2 = 0.693 + 10 = 10.693
  - v5 = v4 - v3 = 10.693 - 0.959 = 11.652
  - y = v5 = 11.652
- $ \frac{\partial f(x1, x2)}{\partial x1} 
    = \frac{\partial v_{-1}}{\partial x1}(\frac{\partial v_1}{\partial v_{-1}}\frac{\partial v_4}{\partial v_1} + \frac{\partial v_2}{\partial v_{-1}}\frac{\partial v_4}{\partial v_2})\frac{\partial v_5}{\partial v_4}\frac{\partial f(x1, x2)}{\partial v_5}
  $
- tangent trace
  - x1
    - $ v'_i = \frac{\partial v_i}{\partial x_1} $
    - $ y'_j = \frac{\partial y_j}{\partial x_1} $

    - $ v'_{-1} = \frac{\partial v_{-1}}{\partial x_1} = 1 $
    - $ v'_0 = \frac{\partial v_0}{\partial x_1} = 0 $
    - $ v'_1 = \frac{\partial v_1}{\partial x_1} = \frac{\partial ln(v_{-1})}{\partial x_1} = \frac{1}{v_{-1}} * v'_{-1} = \frac12 * 1 = \frac12  $
    - $ v'_2 = \frac{\partial v_{-1}v_0}{\partial x_1} = v'_{-1} v_0 + v'_0 v_{-1} = 1*v_0 + 0*2 = v_0 = 5 $
    - $ v'_3 = \frac{\partial sin(v_0)}{\partial x_1} = \frac{\partial sin(v_0)}{\partial v_0} * v'_0 = cos(v_0)*0 = 0 $
    - $ v'_4 = \frac{\partial (v1 + v2)}{\partial x_1} = v'_1 + v'_2 = \frac12 + 5 = 5.5 $
    - $ v'_5 = \frac{\partial (v4 - v3)}{\partial x_1} = v'_4 - v'_3 = 5.5 - 0 = 5.5 $
    - $ y' = \frac{\partial v_5)}{\partial x_1} = v'_5 = 5.5 $

  - x2
    - $ v'_i = \frac{\partial v_i}{\partial x_2} $
    - $ y'_j = \frac{\partial y_j}{\partial x_2} $

    - $ v'_{-1} = \frac{\partial v_{-1}}{\partial x_2} = 0 $
    - $ v'_0 = \frac{\partial v_0}{\partial x_2} = 1 $
    - $ v'_1 = \frac{\partial v_1}{\partial x_2} = \frac{\partial ln(v_{-1})}{\partial x_2} = {\frac{1}{v_{-1}}} * v'_{-1} = /frac12 * 0 = 0 $
    - $ v'_2 = \frac{\partial v_{-1}v_0}{\partial x_2} = v'_{-1} v_0 + v'_0 v_{-1} = 0*5 + 1*2 = 2 $
    - $ v'_3 = \frac{\partial sin(v_0)}{\partial x_2} = \frac{\partial sin(v_0)}{\partial v_0} * v'_0 = cos(5)*1 = 0.284 $
    - $ v'_4 = \frac{\partial (v1 + v2)}{\partial x_2} = v'_1 + v'_2 = 0 + 2 = 2 $
    - $ v'_5 = \frac{\partial (v4 - v3)}{\partial x_2} = v'_4 - v'_3 = 2 - 0.284 = 1.716 $

- reverse trace
  - $ \overline x_1 = \overline v_{-1} = 5.5 $
  - $ \overline x_2 = \overline v_0 = 1.716 $

  - let $ \overline y = \overline v_5 = 1 $
  - $ \overline v_4 
    = \overline v_5 \frac{\partial v_5}{\partial v_4} 
    = \overline v_5 \frac{\partial (v_4 - v_3)}{\partial v_4} 
    = \overline v_5 * 1 = 1
    $
  - $ \overline v_3
    = \overline v_5 \frac{\partial v_5}{\partial v_3} 
    = \overline v_5 \frac{\partial (v_4 - v_3)}{\partial v_3} 
    = \overline v_5 * -1 = -1
    $
  - $ \overline v_2
    = \overline v_4 \frac{\partial v_4}{\partial v_2} 
    = \overline v_4 \frac{\partial (v1 + v2}{\partial v_2} 
    = \overline v_4 * 1 = 1
    $
  - $ \overline v_1
    = \overline v_4 \frac{\partial v_4}{\partial v_1} 
    = \overline v_4 \frac{\partial (v1 + v2}{\partial v_1} 
    = \overline v_4 * 1 = 1
    $
  - $ \overline v_0
    = \overline v_3 \frac{\partial v_3}{\partial v_0} + \overline v_2 \frac{\partial v_2}{\partial v_0}
    = \overline v_3 \frac{\partial sin(v0)}{\partial v_0} + \overline v_2 \frac{\partial v_{-1} v0}{\partial v_0} 
    = \overline v_3 * cos(v_0) + \overline v_2(0*v0 + v_{-1}*1)
    = -1*cos(5) + 1*(0+2*1) = - 0.284 + 2 = 1.716 
    $
  - $ \overline v_{-1}
    = \overline v_1 \frac{\partial v_1}{\partial v_{-1}} + \overline v_2 \frac{\partial v_2}{\partial v_{-1}}
    = \overline v_1 \frac{\partial ln(v_{-1})}{\partial v_{-1}} + \overline v_2 \frac{\partial v_{-1} v0}{\partial v_{-1}} 
    = \overline v_1 * \frac{1}{v_{-1}} + \overline v_2*(v0)
    = 1*\frac12 + 1*5 = 0.5 + 5 = 5.5
    $