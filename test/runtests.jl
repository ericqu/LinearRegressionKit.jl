using LinearRegressionKit
using Test, DataFrames, StatsModels

leaq(a,b) = (a <= b) || (a ≈ b)

# include("test_sweep_operator.jl")
# include("test_utilities.jl")
include("test_LinearRegression.jl") 
# include("test_cooksd.jl")
# include("test_lessthanfullrank.jl")
# include("test_noint.jl")
# include("test_heteroscedasticity.jl")
# include("test_kfold.jl")
# include("test_ridge.jl")
