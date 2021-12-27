## Tutorial ridge regression

This tutorial gives a brief introduction to ridge regression. The tutorial makes use of the acetylene dataset from "Marquardt, D. W., and Snee, R. D. (1975). “Ridge Regression in Practice.” American Statistician 29:3–20.", and the follow the same outline as the [SAS documentation](https://documentation.sas.com/doc/en/statug/15.2/statug_reg_examples05.htm)

### First, creating the dataset.

We create the dataset with the help of the `DataFrames.jl` package.

```@example ridgeregression
using DataFrames
x1 = [1300, 1300, 1300, 1300, 1300, 1300, 1200, 1200, 1200, 1200, 1200, 1200, 1100, 1100, 1100, 1100]
x2 = [7.5, 9.0, 11.0, 13.5, 17.0, 23.0, 5.3, 7.5, 11.0, 13.5, 17.0, 23.0, 5.3, 7.5, 11.0, 17.0]
x3 = [0.012, 0.012, 0.0115, 0.013, 0.0135, 0.012, 0.04, 0.038, 0.032, 0.026, 0.034, 0.041, 0.084, 0.098, 0.092, 0.086]
y = [49.0, 50.2, 50.5, 48.5, 47.5, 44.5, 28.0, 31.5, 34.5, 35.0, 38.0, 38.5, 15.0, 17.0, 20.5, 29.5]
df = DataFrame(x1= x1, x2= x2, x3= x3, y= y)

```

### Second, make a least square regression

We make a ordinary least squares (OLS) regression for comparison.

```@example ridgeregression
using LinearRegressionKit, StatsModels
using VegaLite

f = @formula(y ~ x1 + x2 + x3 + x1 & x2 + x1^2)
lm, ps = regress(f, df, "all", req_stats=["default", "vif"])
lm
```

We observe that the VIF for the coefficients are highs, and hence indicate likely multicollinearity.

### Ridge regression

Ridge regression requires a parameter (k), while there are method to numerically suggest a k. It is also possible to trace the coefficients and VIFs values to let the analyst choose a k. Here we are going to trace for the k between 0. and 0.1 by increment of 0.0005. We display only the results for the first 5 k.

```@example ridgeregression
rdf, ps = ridge(f, df, 0.0:0.0005:0.1, traceplots=true)
rdf[1:5 , :]
```

Here is the default trace plot for the coefficients:
```@example ridgeregression
ps["coefs traceplot"]
```
!!! note
    It is an issue that the `&` in the variable names are replaced by ` ampersand ` because otherwise it would cause issue either on the web display or on the SVG display. This only affect the plot generate with the library. One can directly use the DataFrame to generate its own plots. 

And here is the default trace plot for the VIFs:
```@example ridgeregression
ps["vifs traceplot"]
```

As it is difficult to see the VIFs traces, it is also possible to request the version of hte plot with the y-axis log scaled.
```@example ridgeregression
ps["vifs traceplot log"]
```

Once a `k` has been selected (in this case 0.004) a regular ridge regression with this value can executed.
```@example ridgeregression
rlm = ridge(f, df, 0.004)
```

From there the regular `predict_*` functions can be used, although only the `predicted` and potentially the `residuals` statistics will be calculated.

```@example ridgeregression
res = predict_in_sample(rlm, df)
```
