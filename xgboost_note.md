# Note for XGBoost
[Paper - XGBoost: A Scalable Tree Boosting System](https://arxiv.org/pdf/1603.02754.pdf)

Overview:
+ sec 2 - tree boosting and regularized objective
+ sec 3 - split finding methods
+ sec 4 - experimential results
+ sec 5 - related work
+ sec 6 - end-to-end evaluations
+ sec 7 - final conclusion

# 2. Tree Boosting In a NutShell
## 2.1 Regularized Learning Objective

+ __n__ examples, __m__ features and __K__ additive function
$$
\hat{y_i} = \phi(x_i) = \sum_{k}f_k(x_i),\ \ \  f_k \in\ F \\
where\ F = \{f(x) = w_{q(x)}\}(q:\ R^m \rightarrow T, w \in R^T)
$$
+ __q__ - the structure of each tree that maps an example to the corresponding leaf index
+ __T__ is the number of leaves in the tree 
    + leaf node does not have any child
+ Each $f_k$ corresponds to an independent tree structure __q__ and leaf weights __w__
    + $w_i$ is the score on i-th leaf
+ The __regularized objective__ will be minimized to learn the functions used in the model
$$
L(\phi) = \sum_{i}(\hat{y_i}, y_i) + \sum_{k}(\Omega(f_k))\\

where\ \Omega(f) = \gamma T + \frac{1}{2}\lambda||w||^2
$$

## 2.2 Gradient Tree Boosting


