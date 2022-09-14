# bagging
- random tree
- training
  - same train dataset
  - different models
- inference
  - same test dataset
  - result = weighted averge

# boosting
## paper
- XGBoost: A Scalable Tree Boosting System
- https://dl.acm.org/doi/pdf/10.1145/2939672.2939785
  - optimize

# the difference
| | bagging | boosting |
| --- | --- | --- |
| models | great enough | not good enough|
| models | leverages unstable base learners that we weak because of <font color=#ff000> overfitting </font> | leverages stable base learners that we weak because of <font color=#ff000> underfitting </font> |
| member | random_tree | adaboost(weighted sample), gbdt, xgboost |
| training | training all models at the same time | training models one by one |
| predict | weighted average | summary |
| pros | | parallel & efficient; parameter configable; tha actual effect is good |

  
# boost tree
<font color=#ff000> training based on residual </font>
- abbreviation
  - tds = training dataset
  - p = predict
  - r = residual
- model1
  - tds_1 = tds
  - p_1 from model1 base on tds_1
  - r_1 = target - p_1
- model2
  - tds_2 = r_1
  - p_2 from model2 base on tds_2
  - r_2 = target - p_1 - p_2
- model3
  - tds_3 = r_2
  - predict_3 from model3 base on tds_3
  - r_3 = target - p_1 - p_2 - p_3
- ...
- <font color=#ff000> how to create the model in time ? </font>

# algorithm
<font color=#ff000> **loss -> taylor -> paramerize -> search** </font>

- create object function
  - loss function
  - definition the problem
- optimize 
  - taylor expression
    - simplify
    - LR->SGD
    - object functio in xgboost is distributed
  - describe tree in object function
  - greedy search

# create model
- K tree(model)
  $$ y_i = \sum_{k=1}^{K}f_k(x_i),f_k \in F $$
  - k is the number of tree
  - xi is dataset
  - f : feasible place
  - f1(xi) is the predict from model_k based on xi
  - yi predict value
- object function
  $$ obj = \sum_{i=1}^nl(y_i,y_i') + \sum_{k=1}^K\Omega(f_k) $$
  - loss
    - differentiable convex loss funstion
    - MSE, cross-entropy, GFLOPS
    - yi target
    - yi' prediction
  - penalty
    - penalizes the complexity
    - regression tree functions
    - avoid over-fitting, L2-regularization
- minimize

# complex of tree
- the number of leaf point
- the deep of tree
- the value of leaf point

# Additive Training
- K trees
- predict value
  - yi(1),yi(2),...,yi(k-1)
- base: xi
- yi(0) = 0, default case
- yi(1) = f1(xi) = yi(0) + f1(xi)
- yi(2) = f1(xi) + f2(xi) = yi(1) + f2(xi)
- ...
- yi(k) = f1(xi) + f2(xi) + ... + fk(xi) = yi(k-1) + fk(xi)
= $$ \sum_{j=1}^{K-1}f_j(x_i) + f_K(x_i) $$
- simplify
$$ obj = \sum_{i=1}^nl(y_i,y_i') + \sum_{k=1}^K\Omega(f_k) $$
$$ = \sum_{i=1}^nl(y_i,y_i^{(k-1)}+f_K(x_i)) + \sum_{j=1}^{K-1}\Omega(f_j) + \Omega(f_K) $$
- constant $$ \sum_{j=1}^{K-1}\Omega(f_j) $$
- minimize $$ \sum_{i=1}^nl(y_i,y_i^{(k-1)}+f_K(x_i)) + \Omega(f_K) $$
  
  - yi is target
  - yi(K-1) is predict value after K-1 tree
  - f_K(x_i) current oredict
  - Omega(f_K) current complexity

# taylor expression
- taylor
$$ f(x+\Delta x) \approx f(x) + f'(x)\Delta x + \frac12 f''(x) \Delta x^2 $$
- obj
$$ obj = \sum_{i=1}^nl(y_i,y_i^{(k-1)}+f_K(x_i)) + \Omega(f_K) $$
$$ f(x) = l(y_i,y_i^{(k-1)}) $$
$$ f(x+\Delta x) = l(y_i,y_i^{(k-1)} + f_k^(x_i)) $$
$$ obj = \sum_{i=1}^n[l(y_i,y_i^{(k-1)}) + \vartheta l(y_i,y_i^{(k-1)}) f_k(x_i) + \frac12 \vartheta ^2 l(y_i,y_i^{(k-1)}) f_k^2(x_i)] + \Omega(f_K) ]$$
$$ = \sum_{i=1}^n[l(y_i,y_i^{(k-1)}) + g_i f_k(x_i) + \frac12 h_i f_k^2(x_i)] + \Omega(f_K) ]$$
minimize 
$$ = \sum_{i=1}^n[g_i f_k(x_i) + \frac12 h_i f_k^2(x_i)] + \Omega(f_K) ]$$
constant
  - residual
    - g_i
    - h_i
  - $$l(y_i,y_i^{(k-1)})$$

# parameterize
Assume we got the tree shape, we can calculate the minimize value of current tree through second-order Taylor expansion

- tree: $$ f_k(x_i) $$
- complexity: $$ \Omega(f_K) $$
## tree definition
- w
  - vector of leaf value
  - ordered from left to right
- q(x)
  - location of x
- example
  - w = (w1, s2, w3)
  - sample x1-x5
  - q(x1) = 1, x1 in w1
  - q(x2) = 3, x2 in w3
  - q(x3) = 1, x1 in w1
  - q(x4) = 2, x1 in w2
  - q(x5) = 3, x1 in w3
- Ij = {i|q(xi)=j}
  - I1 = {1,3}
  - I2 = {4}
  - I3 = {2,5}

## tree's complexity
- T
  - the number of leaf node
- W
  - leaf value
$$ \Omega (f_k) = \Gamma T + \frac12 \lambda \sum_{j=1}^T w_j^2 $$

## new objective function

$$ obj = \sum_{i=1}^n[g_i f_k(x_i) + \frac12 h_i f_k^2(x_i)] + \Omega(f_K) ]$$
$$ = \sum_{i=1}^n[g_i w_{q(x_i)} + \frac12 h_i w_{q(x_i)}^2] + \Gamma T + \frac12 \lambda \sum_{j=1}^T w_j^2$$

## the optimal solution
- leaf_node_1 $$ g_1 w_{q(x_1)} + g_3 w_{q(x_3)} $$
- leaf_node_2 $$ g_4 w_{q(x_4)} $$
- leaf_node_3 $$ g_2 w_{q(x_2)} + g_5 w_{q(x_5)} $$

$$ = \sum_{j=1}^T [\sum_{i \in I_j}g_j w_j + \frac12 (\sum_{i \in I_j}h_j + \lambda ^2)] w^2_j + \Gamma T $$ 

$$ = a w_j + b w^2_j $$
$$ w_j^* = -\frac b{2a} $$
$$ w_j^* = -\frac {G_i}{H_j + \lambda} $$
$$ obj^* = -\frac12\sum_{j=1}^T \frac {G_i^2}{H_j + \lambda} + \Gamma T $$ 

# Search tree
- example: 1000 tree
$$ \min {\{obj_k^{x(i)}\}}_{i=1}^{1000} $$

- brute ferce search = force search
  - too many
- greedy

## decision tree
- root {1,2,3,4,5,6,7,8}
- feature_1
  - {1,3,4}
  - {2,5,6,7,8}
  - score = - InformationGain
    - big entropy decrease uncertainty
    - maximum IG
- score = Entropy_before - Entropy_after = ObjB - ObjA
  - find out which feature will get the max entropy

## split
- feature_N
- partitional region

$$ obj^* = -\frac12\sum_{j=1}^T \frac {G_i^2}{H_j + \lambda} + \Gamma T $$

## example
- feature_1
  - T = 2
    - {7,8}
    - {1,2,3,4,5,6}
$$ obj^*_{before} = -\frac12[\frac {{(g_7+g_8)}^2}{h_7+h_8+\lambda} + \frac {{(g_1+...+g_6)}^2}{h_1+...+h_6+\lambda}] + \Gamma *2 $$ 
- feature_2
  - T = 3
    - {7,8}
    - {1,3,5}
    - {2,4,6}
$$ obj^*_{after} = -\frac12[\frac {{(g_7+g_8)}^2}{h_7+h_8+\lambda} + \frac {{(g_1+g_3+g_5)}^2}{h_1+h_3+h_5+\lambda} + \frac {{(g_2+g_4+g_6)}^2}{h_2+h_4+h_6+\lambda}] + \Gamma *3 $$ 

$$ obj^*_{before} - obj^*_{after} = \frac12[\frac {{g_l}^2}{h_l+\lambda} - \frac {{g_r}^2}{h_r+\lambda} - \frac {{(g_l+g_r)}^2}{h_l+h_r+\lambda}] - \Gamma $$
where:
$$g_l = g_1+g_3+g_5$$
$$h_l = h_1+h_3+h_5$$
$$g_r = g_2+g_4+g_6$$
$$h_r = h_2+h_4+h_6$$







