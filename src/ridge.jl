# # Ridge regression adapted from the SAS fomulas   
# # see https://blogs.sas.com/content/iml/2013/03/20/compute-ridge-regression.html

"""
    function ridge(f::StatsModels.FormulaTerm, df::DataFrames.AbstractDataFrame, k::Float64 ; 
    weights::Union{Nothing,String}=nothing, remove_missing=false, contrasts=nothing)

    Ridge regression, expects a k parameter (also known as k).
    When weights are provided, result in a weighted ridge regression.
"""
function ridge(f::StatsModels.FormulaTerm, df::DataFrames.AbstractDataFrame, k::Float64 ; 
        weights::Union{Nothing,String}=nothing, remove_missing=false, contrasts=nothing)
    X, y, n, p, intercept, f, copieddf, updatedformula, isweighted, dataschema = 
    design_matrix!(f, df, weights=weights, remove_missing=remove_missing, contrasts=contrasts, ridge=true)

    cweights = nothing
    if !isnothing(weights)
        cweights = copieddf[!, weights]
    end
    coefs, vifs = iridge(X, y, intercept, k, cweights)

    mse, rmse, r2, adjr2 = iridge_stats(X, y, coefs, intercept, n, p, cweights)

    res_ridge_reg = ridgeRegRes(
        k, p, n, intercept, coefs,
        vifs, mse, rmse, r2, adjr2, f, updatedformula, dataschema, isweighted, weights)

    return res_ridge_reg
end

"""
    function ridge(f::StatsModels.FormulaTerm, df::DataFrames.AbstractDataFrame, ks::AbstractRange ; 
    weights::Union{Nothing,String}=nothing, remove_missing=false, contrasts=nothing, traceplots=false)

    Ridge regression, expects a range of k parameter (also known as k).
    When weights are provided, result in a weighted ridge regression.
    When traceplots are requested, also return a dictionnary of trace plots.
"""
function ridge(f::StatsModels.FormulaTerm, df::DataFrames.AbstractDataFrame, ks::AbstractRange ;
            weights::Union{Nothing,String}=nothing, remove_missing=false, contrasts=nothing,
            traceplots = false )
    X, y, n, p, intercept, f, copieddf, updatedformula, isweighted, dataschema = 
    design_matrix!(f, df, weights=weights, remove_missing=remove_missing, contrasts=contrasts, ridge=true)

    vcoefs, vvifs = iridge(X, y, intercept, ks)
    cweights = nothing
    if !isnothing(weights)
        cweights = copieddf[!, weights]
    end
    vcoefs, vvifs = iridge(X, y, intercept, ks, cweights)

    vmse = Vector{Float64}(undef, length(ks))
    vrmse = Vector{Float64}(undef, length(ks))
    vr2 = Vector{Float64}(undef, length(ks))
    vadjr2 = Vector{Float64}(undef, length(ks))

    for (i, x) in enumerate(ks)
        vmse[i], vrmse[i], vr2[i], vadjr2[i] = iridge_stats(X, y, vcoefs[i], intercept, n, p, cweights)
    end

    coefs_names = encapsulate_string(string.(StatsBase.coefnames(updatedformula.rhs)))
    vifs_names = "vif " .* coefs_names
    cv = [ks vmse vrmse vr2 vadjr2 transpose(hcat(vcoefs...)) transpose(hcat(vvifs...)) ]
    all_names = ["k", "MSE", "RMSE", "R2", "ADJR2", coefs_names..., vifs_names... ]
    df = DataFrame(cv, all_names)

    if traceplots == false
        return df 
    else
        return df, ridge_traceplots(df)
    end

end

"""
    function prepare_ridge(X_orig, y_orig, intercept, weights::Union{Nothing,Vector{Float64}}=nothing)

    (internal) Prepare the design matrix for ridge regression by centering the data (potentially weighted).
"""
function prepare_ridge(X_orig, y_orig, intercept, weights::Union{Nothing,Vector{Float64}}=nothing)
    X = deepcopy(X_orig)
    y = deepcopy(y_orig)

    # removes the intercept (assumed to be the first column)
    if intercept
        X = X[:, deleteat!(collect(axes(X, 2)), 1)]
    end

    Xmeans = nothing
    ymean = nothing
    if !isnothing(weights)
        Xmeans = mean(X, aweights(weights), dims=1)
        ymean = mean(y, aweights(weights))
    else
        # get the means the Xs and ys
        Xmeans = mean(X, dims=1)
        ymean = mean(y)
    end

    # center the X and y
    for i in 1:size(X, 2)
        X[:, i] .-= Xmeans[i]
    end
    y .-= ymean

    # if needed apply weights to the centered X and y
    if !isnothing(weights)
        X = X .* sqrt.(weights)
        y = y .* sqrt.(weights)
    end

    XTX = X'X
    D = Diagonal(diag(XTX))
    Z = X / sqrt(D)
    ZTZ = Z'Z 

    return XTX, D, Z, ZTZ, ymean, Xmeans, y, X
end

"""
    function iridge(X_orig, y_orig, intercept, k::Float64, weights::Union{Nothing,Vector{Float64}}=nothing)
    XTX, D, Z, ZTZ, ymean, Xmeans, y, X = prepare_ridge(X_orig, y_orig, intercept, weights)

    (internal) compute the coefficient(s) and the VIF of a ridge regression given a scalar k.
"""
function iridge(X_orig, y_orig, intercept, k::Float64, weights::Union{Nothing,Vector{Float64}}=nothing)
    XTX, D, Z, ZTZ, ymean, Xmeans, y, X = prepare_ridge(X_orig, y_orig, intercept, weights)

    invZTZ = pinv(ZTZ + k * I)
    coefs = invZTZ * (Z' * y) ./ (sqrt.(diag(XTX)))
    vifs = diag(invZTZ * ZTZ * invZTZ)
    
    if (intercept)
        # get intercept back
        interceptvalue = ymean - sum(vec(Xmeans) .* coefs)
        coefs = vec([interceptvalue coefs...])
        vifs = vec([0. vifs...])
    end

    return coefs, vifs
end

"""
    function iridge(X_orig, y_orig, intercept, ks::AbstractRange, weights::Union{Nothing,Vector{Float64}}=nothing)
    XTX, D, Z, ZTZ, ymean, Xmeans, y, X = prepare_ridge(X_orig, y_orig, intercept, weights)

    (internal) compute the coefficient(s) and the VIF for each ridge regression with a range of k.

"""
function iridge(X_orig, y_orig, intercept, ks::AbstractRange, weights::Union{Nothing,Vector{Float64}}=nothing)
    XTX, D, Z, ZTZ, ymean, Xmeans, y, X = prepare_ridge(X_orig, y_orig, intercept, weights)

    vcoefs = Vector{Vector{}}(undef, length(ks))
    vvifs = Vector{Vector{}}(undef, length(ks))

    for (i, k) in enumerate(ks)
        invZTZ = pinv(ZTZ + k * I)
        vcoefs[i] = invZTZ * (Z' * y) ./ (sqrt.(diag(XTX)))
        vvifs[i] = diag(invZTZ * ZTZ * invZTZ)
        
        if (intercept)
            # get intercept back
            interceptvalue = ymean - sum(vec(Xmeans) .* vcoefs[i])
            vcoefs[i] = vec([interceptvalue vcoefs[i]...])
            vvifs[i] = vec([0. vvifs[i]...])
        end
    end

    return vcoefs, vvifs
end

"""
    function iridge_stats(X, y, coefs, intercept, n, p, weights::Union{Nothing,Vector{Float64}}=nothing)

    (internal) compute the limited stats from a ridge regression.
"""
function iridge_stats(X, y, coefs, intercept, n, p, weights::Union{Nothing,Vector{Float64}}=nothing)
    ŷ = lr_predict(X, coefs, intercept)
    residuals = y .- ŷ
    sse = nothing 
    if isnothing(weights)
        sse = sum(residuals.^2)
    else
        sse = sum(residuals.^2, aweights(weights))
    end

    mse = sse / (n - p)
    rmse = real_sqrt(mse)

    sst = nothing
    if isnothing(weights)
        sst = getSST(y, intercept)
    else
        sst = getSST(y, intercept, weights, true)
    end
    r2 = 1. - (sse / sst)
    adjr2 = 1. - ((n - convert(Int64, intercept)) * (1. - r2)) / (n - p)

    return mse, rmse, r2, adjr2
end

"""
Store the result of a single ridge (potentially weighted) regression
"""
struct ridgeRegRes
    k::Float64
    p::Float64
    observations
    intercept::Bool
    coefs::Vector
    VIF::Vector
    MSE::Float64
    RMSE::Float64
    R2::Float64
    ADJR2::Float64
    modelformula
    updatedformula
    dataschema
    weighted::Bool
    weights::Union{Nothing,String}
end

"""
    function Base.show(io::IO, rr::ridgeRegRes) 

    Display information about the fitted ridge regression model
"""
function Base.show(io::IO, rr::ridgeRegRes) 
    if rr.weighted
        println(io, "Weighted Ridge regression")
    else
        println(io, "Ridge Regression")
    end
    println(io, "Constant k:\t", rr.k)
    println(io, "Model definition:\t", rr.modelformula)
    println(io, "Used observations:\t", rr.observations)
    println(io, "Model statistics:")
    @printf(io, "  R²: %g\t\t\tAdjusted R²: %g\n", rr.R2, rr.ADJR2)
    @printf(io, "  MSE: %g\t\t\tRMSE: %g\n", rr.MSE, rr.RMSE)

    helper_print_table(io, "Coefficients statistics:", 
        [rr.coefs, rr.VIF], ["Coefs", "VIF"], rr.updatedformula)

end

"""
    function predict_in_sample(rr::ridgeRegRes, df::AbstractDataFrame; dropmissingvalues=true)

    Using the estimated coefficients from the regression make predictions, and calculate related statistics.
"""
function predict_in_sample(rr::ridgeRegRes, df::AbstractDataFrame; dropmissingvalues=true)
    predict_internal(df, rr.modelformula, rr.updatedformula, rr.weighted, rr.weights, nothing, rr.coefs, rr.intercept,
        nothing, nothing, nothing, nothing, rr.p, rr.observations, false;
        α=nothing, req_stats=[:predicted, :residuals], dropmissingvalues=dropmissingvalues)
end

"""
    function predict_out_of_sample(rr::ridgeRegRes, df::AbstractDataFrame; dropmissingvalues=true)

    Similar to `predict_in_sample` although it does not expect a response variable nor produce statistics requiring a response variable.
"""
function predict_out_of_sample(rr::ridgeRegRes, df::AbstractDataFrame; dropmissingvalues=true)
    predict_internal(df, rr.modelformula, rr.updatedformula, rr.weighted, rr.weights, nothing, rr.coefs, rr.intercept,
        nothing, nothing, nothing, nothing, rr.p, rr.observations, true;
        α=nothing, req_stats=[:predicted], dropmissingvalues=dropmissingvalues)
end
