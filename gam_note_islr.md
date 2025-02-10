# Note of GAM (Chpt 7 Moving Beyond Linearity from ISLR)

Overview

+ Polynominal  Regression
    + a simple way to provide a non-linear fit to data
    + extends the linear model by adding extra predictors, obtained by raising each of the original predictors to a power
+ Step Functions
    + cut the range of a variable into $K$ distinct regious
+ Regression Splines
    + a combination of polynominals and step functions
    + dividing the range of $X$ into $K$ distinct regions, and fit the data to polynominals within each region
    + polynominals are joined smoothly at the region boundaries (__knots__)
+ Smoothing Splines
    + similar to regression
    + result from minimizing a residual sum of squares criterion subject to a smoothness penalty
+ Local Regression
    + also similar to regression
    + regions are allow to overlap
+ Generalized Addictive Models
    + extend the methods above to deal with multiple predictors

# Section 1 - Polynominal Regression

A polynominal function representation is shown below. It could also be considered as a standard linear model with predictors $x_i$, $x_i^2$, ..., $x_i^d$ estimated using least squares linear regression
$$
y_i = \beta_0 + \beta_1x_i + \beta_2x_i^2 + ... + \beta_dx_i^d + \epsilon_i
$$

# Section 3 - Basis Functions

Polynominal and piecewise-constant regression models are special cases of a base function approach. For fixed and known  basis functions $b_1(·)$, $b_2(·)$, ..., $b_K(·)$ applied to a variable $X$:

$$
y_i = \beta_0 + \beta_1b_1(x_i) + \beta_2b_2(x_i) + ... + \beta_Kb_K(x_i) + \epsilon_i
$$

For polynominal functions, $b_j(x_i) = x_i^j$, for piecewise functions $b_j(x_i) = I(cj \leq x_i \leq c_{j+1})$ 