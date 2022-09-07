# GAM (Generalized Addictive Model)

+ [GAM Wiki](https://en.wikipedia.org/wiki/Generalized_additive_model)
+ [Backfitting Algorithm](https://en.wikipedia.org/wiki/Backfitting_algorithm)
+ [Restricted Maximum Likelihood](https://en.wikipedia.org/wiki/Restricted_maximum_likelihood)

# 1. About GAM

It is a generalized linear model in which the linear response variable ($Y$) depends linearly on several unknown smoothing functions of some predictors varables.

The structure of the model is like

$$
g(\mu_i) = X_i^*\theta +  f_1(x_{1i}) + f_2(x_{2i}) + f_3(x_{3i}, x_{4i}) + ... \\
$$

where

$$
\mu_i = E(Y_i) \text{ and } Y_i \text{ is some exponential family distribution}
$$

+ $Y_i$ is a response variable
+ $X_i^*$ is a row of the model matrix for any strictly parametric  model components
+ $\theta$ is the corresponding parameter vector
+ $f_j$ are smooth functions of covariates $x_k$

## 1.1 Univariate smooth functions

Starting with creating a model with one smooth function of one covariate which 
$$
y_i = f(x_i) + \epsilon_i
$$
In this case $x_i$ is the only covariate and $f$ is a smooth function and $\epsilon_i$ are i.i.d $N(0, \sigma^2)$ r.v

We can represent the smooth function as regression splines by __choosing polynomial basis functions__ $f(x) = \sum\limits_{i = 1}^{q}b_i(x)\beta_i$. The $\beta_i$ are some values of unknown parameters.

If we consider 3th order polynomial, in this way $b_1(x) = 1, b_2(x) = x, b_3(x) = x^2, b_4(x) = x^3$. The representation then becomes $f(x) = \beta_1 + x\beta_2, x^2\beta_3 + x^3\beta_4$

## 1.2 Splines and Knots
[reference in stackoverflow](https://stats.stackexchange.com/questions/517375/splines-relationship-of-knots-degree-and-degrees-of-freedom)

Splines are piecewise polynominals joined at the points called knots.
+ Degree - the degree of the polynominals
+ Degrees of Freedoms (df) - how many parameters you have to estimate

The higher the degrees of freedoms, the wigglier the spline gets because the number of knots is increased.


# 2. Backfitting Algorithm


