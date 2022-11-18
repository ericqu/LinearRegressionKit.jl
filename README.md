[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://ericqu.github.io/LinearRegressionKit.jl/dev/)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://ericqu.github.io/LinearRegressionKit.jl/stable)
# LinearRegressionKit.jl

LinearRegressionKit.jl implements linear regression using the least-squares algorithm (relying on the sweep operator). This package is in the beta stage. Hence it is likely that some bugs exist. Furthermore, the API might change in future versions. User's or prospective users' feedback is welcome.

# Installation
Enter the Pkg REPL by pressing ] from the Julia REPL. Then install the package with: 
``` pkg> add LinearRegressionKit ``` or  ```pkg> add https://github.com/ericqu/LinearRegressionKit.jl.git ```. 
To uninstall use ```  pkg> rm LinearRegressionKit```

# Usage

The following is a simple usage:

```julia 
using LinearRegressionKit, DataFrames, StatsModels

x = [0.68, 0.631, 0.348, 0.413, 0.698, 0.368, 0.571, 0.433, 0.252, 0.387, 0.409, 0.456, 0.375, 0.495, 0.55, 0.576, 0.265, 0.299, 0.612, 0.631]
y = [15.72, 14.86, 6.14, 8.21, 17.07, 9.07, 14.68, 10.37, 5.18, 9.36, 7.61, 10.43, 8.93, 10.33, 14.46, 12.39, 4.06, 4.67, 13.73, 14.75]

df = DataFrame(y=y, x=x)

lr = regress(@formula(y ~ 1 + x), df)

```

which outputs the following information:
```
Model definition:       y ~ 1 + x
Used observations:      20
Model statistics:
  R²: 0.938467                  Adjusted R²: 0.935049
  MSE: 1.01417                  RMSE: 1.00706
  σ̂²: 1.01417
  F Value: 274.526 with degrees of freedom 1 and 18, Pr > F (p-value): 2.41337e-12
Confidence interval: 95%

Coefficients statistics:
Terms ╲ Stats │       Coefs      Std err            t     Pr(>|t|)         code       low ci      high ci
──────────────┼──────────────────────────────────────────────────────────────────────────────────────────
(Intercept)   │    -2.44811     0.819131     -2.98867     0.007877          **      -4.16904    -0.727184
x             │     27.6201      1.66699      16.5688  2.41337e-12         ***       24.1179      31.1223

        Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
```

# Contrasts with Julia Stats GLM package
First, the GLM package provides more than linear regression with Ordinary Least-Squares through the Generalized Linear Model with Maximum Likelihood Estimation.

LinearRegressionKit accepts model without intercept. Like models made with GLM the intercept is implicit, and to enable the no intercept the user must specify it in the formula (for instance ```y  ~ 0 + x```).

LinearRegressionKit supports analytical weights; GLM supports frequency weights.

Both LinearRegressionKit and GLM rely on StatsModels.jl for the model's description (@formula); hence it is easy to move between the two packages. Similarly, contrasts and categorical variables are defined in the same way facilitating moving from one to the other when needed.

LinearRegressionKit relies on the Sweep operator to estimate the coefficients, and GLM depends on Cholesky and QR factorizations.

The Akaike information criterion (AIC) is calculated with the formula relevant only for Linear Regression hence enabling comparison between linear regressions (AIC=n log(SSE / n) + 2p; where SSE is the Sum of Squared Errors and p is the number of predictors). On the other hand, the AIC calculated with GLM is more general (based on log-likelihood), enabling comparison between a broader range of models.

LinearRegressionKit package provides access to some robust covariance estimators (for Heteroscedasticity: White, HC0, HC1, HC2 and HC3 and for HAC: Newey-West)

Ridge Regression (potentially with analytical weights) is implemented in the LinearRegressionKit package.

# List of Statistics 
## List of Statistics calculated about the linear regression model:
- AIC: Akaike information criterion with the formula AIC=n log(SSE / n) + 2p; where SSE is the Sum of Squared Errors and p is the number of predictors.
- SSE Sum of Squared Errors as the output from the sweep operator.
- SST as the Total Sum of Squares as the sum over all squared differences between the observations and their overall mean.
- R² as 1 - SSE/SST.
- Adjusted R².
- σ̂² (sigma) Estimate of the error variance.
- Variance Inflation Factor.
- CI the confidence interval based the \alpha default value of 0.05 giving the 95% confidence interval.
- The t-statistic.
- The mean squared error.
- The root of the mean squared error.
- The standard errors and their equivalent with a Heteroscedasticity or HAC covariance estimator
- The t values and their equivalent with a Heteroscedasticity or HAC covariance estimator
- P values for each predictor and their equivalent with a Heteroscedasticity or HAC covariance estimator
- Type 1 & 2 Sum of squares
- Squared partial correlation coefficient, squared semi-partial correlation coefficient. 
- PRESS as the sum of square of predicted residuals errors
- F Value (SAS naming) F Statistic (R naming) is presented with its p-value

## List of Statistics about the predicted values:
- The predicted values
- The residuals values (as the actual values minus the predicted ones)
- The Leverage or the i-th diagonal element of the projection matrix.
- STDI is the standard error of the individual predicted value.
- STDP is the standard error of the mean predicted value
- STDR is the standard error of the residual
- Student as the studentized residuals also knows as the Standardized residuals or internally studentized residuals.
- Rstudent is the studentized residual with the current observation deleted.
- LCLI is the lower bound of the confidence interval for the individual prediction.
- UCLI is the upper bound of the confidence interval for the individual prediction.
- LCLP is the lower bound of the confidence interval for the expected (mean) value.
- UCLP is the upper bound of the confidence interval for the expected (mean) value.
- Cook's Distance
- PRESS as predicted residual errors

# Questions and Feedback
Please post your questions, feedabck or issues in the Issues tabs. As much as possible, please provide relevant contextual information.

# Credits and additional information
- Goodnight, J. (1979). "A Tutorial on the SWEEP Operator." The American Statistician.
- Gordon, R. A. (2015). Regression Analysis for the Social Sciences. New York and London: Routledge.
- https://blogs.sas.com/content/iml/2021/07/14/performance-ls-regression.html
- https://github.com/joshday/SweepOperator.jl
- http://hua-zhou.github.io/teaching/biostatm280-2019spring/slides/12-sweep/sweep.html
- https://github.com/mcreel/Econometrics for the Newey-West implementation
- https://blogs.sas.com/content/iml/2013/03/20/compute-ridge-regression.html
- Code from StatsModels https://github.com/JuliaStats/StatsModels.jl/blob/master/test/extension.jl (in December 2021)

# Examples

The following is a short example illustrating some statistics about the predicted data.
First, a simulation of some data with a polynomial function.

```julia 
using LinearRegressionKit, DataFrames, StatsModels
using Distributions # for the data generation with Normal() and Uniform()
using VegaLite

# Data simulation
f(x) = @. (x^3 + 2.2345x - 1.2345 + rand(Normal(0, 20)))
xs = [x for x in -2:0.1:8]
ys = f(xs)
vdf = DataFrame(y=ys, x=xs)
```
Then we can make the first model and look at the results:

```julia 
lr, ps = regress(@formula(y ~ 1 + x), vdf, "all", 
    req_stats=["default", "vif", "AIC"], 
    plot_args=Dict("plot_width" => 200))
lr
```
```
Model definition:       y ~ 1 + x
Used observations:      101
Model statistics:
  R²: 0.758985                  Adjusted R²: 0.75655
  MSE: 5660.28                  RMSE: 75.2348
  σ̂²: 5660.28                   AIC: 874.744
  F Value: 311.762 with degrees of freedom 1 and 99, Pr > F (p-value): 2.35916e-32
Confidence interval: 95%

Coefficients statistics:
Terms ╲ Stats │       Coefs      Std err            t     Pr(>|t|)         code       low ci      high ci          VIF
──────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────
(Intercept)   │    -26.6547      10.7416     -2.48145    0.0147695           *      -47.9683     -5.34109          0.0
x             │     45.3378      2.56773      17.6568  2.35916e-32         ***       40.2429      50.4327          1.0

        Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
```
This is okay, so let's further review some diagnostic plots.

```julia
[[ps["fit"] ps["residuals"]]
    [ps["histogram density"] ps["qq plot"]]]
```
![Illustrative Overview Plots](https://github.com/ericqu/LinearRegressionKit.jl/raw/main/assets/asset_exe_072_01.svg "Illustrative Overview Plots")

Please note that for the fit plot, the orange line shows the regression line, in dark grey the confidence interval for the mean, and in light grey the interval for the individuals predictions.

Plots are indicating the potential presence of a polynomial component. Hence one might try to add one by doing the following:

```julia 
lr, ps = regress(@formula(y ~ 1 + x^3 ), vdf, "all", 
    req_stats=["default", "vif", "AIC"], 
    plot_args=Dict("plot_width" => 200 ))
```
Giving:
```
Model definition:       y ~ 1 + :(x ^ 3)
Used observations:      101
Model statistics:
  R²: 0.984023                  Adjusted R²: 0.983861
  MSE: 375.233                  RMSE: 19.3709
  σ̂²: 375.233                   AIC: 600.662
  F Value: 6097.23 with degrees of freedom 1 and 99, Pr > F (p-value): 9.55196e-91
Confidence interval: 95%

Coefficients statistics:
Terms ╲ Stats │       Coefs      Std err            t     Pr(>|t|)         code       low ci      high ci          VIF
──────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────
(Intercept)   │  -0.0637235      2.38304   -0.0267404     0.978721                   -4.7922      4.66475          0.0
x ^ 3         │     1.05722    0.0135394      78.0847  9.55196e-91         ***       1.03036      1.08409          1.0

        Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
```
![Illustrative Overview Plots](https://github.com/ericqu/LinearRegressionKit.jl/raw/main/assets/asset_exe_072_02.svg "Illustrative Overview Plots")

Further, in addition to the diagnostic plots helping confirm if the residuals are normally distributed, a few tests can be requested:

```julia
# Data simulation
f(x) = @. (x^3 + 2.2345x - 1.2345 + rand(Uniform(0, 20)))
xs = [x for x in -2:0.001:8]
ys = f(xs)
vdf = DataFrame(y=ys, x=xs)

lr = regress(@formula(y ~ 1 + x^3 ), vdf, 
    req_stats=["default", "vif", "AIC", "diag_normality"])
```
Giving:
```
Model definition:       y ~ 1 + :(x ^ 3)
Used observations:      10001
Model statistics:
  R²: 0.99795                   Adjusted R²: 0.99795
  MSE: 43.4904                  RMSE: 6.59472
  σ̂²: 43.4904                   AIC: 37731.2
  F Value: 4.868e+06 with degrees of freedom 1 and 9999, Pr > F (p-value): 0
Confidence interval: 95%

Coefficients statistics:
Terms ╲ Stats │       Coefs      Std err            t     Pr(>|t|)         code       low ci      high ci          VIF
──────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────
(Intercept)   │     11.3419    0.0816199       138.96          0.0         ***       11.1819      11.5019          0.0
x ^ 3         │     1.04021  0.000471459      2206.35          0.0         ***       1.03928      1.04113          1.0

        Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

Diagnostic Tests:

Kolmogorov-Smirnov test (Normality of residuals):
  KS statistic: 3.05591    observations: 10001    p-value: 0.0
  with 95.0% confidence: reject null hyposthesis.
Anderson–Darling test (Normality of residuals):
  A² statistic: 25.508958    observations: 10001    p-value: 0.0
  with 95.0% confidence: reject null hyposthesis.
Jarque-Bera test (Normality of residuals):
  JB statistic: 240.520153    observations: 10001    p-value: 0.0
  with 95.0% confidence: reject null hyposthesis.
```

Here is how to request the robust covariance estimators:

```julia 
lr = regress(@formula(y ~ 1 + x^3 ), vdf, cov=["white", "nw"])
```
Giving:
```
Model definition:       y ~ 1 + :(x ^ 3)
Used observations:      10001
Model statistics:
  R²: 0.99795                   Adjusted R²: 0.99795
  MSE: 43.4904                  RMSE: 6.59472
  PRESS: 435034
  F Value: 4.868e+06 with degrees of freedom 1 and 9999, Pr > F (p-value): 0
Confidence interval: 95%

White's covariance estimator (HC0):
Terms ╲ Stats │       Coefs      Std err            t     Pr(>|t|)         code       low ci      high ci
──────────────┼──────────────────────────────────────────────────────────────────────────────────────────
(Intercept)   │     11.3419    0.0828903       136.83          0.0         ***       11.1794      11.5044
x ^ 3         │     1.04021  0.000471604      2205.67          0.0         ***       1.03928      1.04113

        Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

Newey-West's covariance estimator:
Terms ╲ Stats │       Coefs      Std err            t     Pr(>|t|)         code       low ci      high ci
──────────────┼──────────────────────────────────────────────────────────────────────────────────────────
(Intercept)   │     11.3419     0.158717        71.46          0.0         ***       11.0308       11.653
x ^ 3         │     1.04021  0.000863819      1204.19          0.0         ***       1.03851       1.0419

        Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
```

Finally if you would like more examples I encourage you to go to the documentation as it gives a few more examples. 

## Notable changes since version 0.76
- Added the F Value (F Statistics) as a default statistic computed when a model is fitted.
- Significance codes similar to R (lm) are also displayed when p_values are requested (which they are by default).