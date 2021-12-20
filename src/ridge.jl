# # Ridge regression adapted from the SAS fomulas   
# # see https://blogs.sas.com/content/iml/2013/03/20/compute-ridge-regression.html

"""
    function ridge(f::StatsModels.FormulaTerm, df::DataFrames.AbstractDataFrame, λ::Float64 ; 
    weights::Union{Nothing,String}=nothing, remove_missing=false, contrasts=nothing )

    Ridge regression, expects a λ parameter (also known as k).
"""
    function ridge(f::StatsModels.FormulaTerm, df::DataFrames.AbstractDataFrame, λ::Float64 ; 
            weights::Union{Nothing,String}=nothing, remove_missing=false, contrasts=nothing )
            # weights will be applied later 
        X, y, n, p, intercept, f, copieddf, updatedformula, isweighted, dataschema = 
        design_matrix!(f, df, weights=weights, remove_missing=remove_missing, contrasts=contrasts, ridge=true)

        cweights = nothing
        if !isnothing(weights)
            cweights = copieddf[!, weights]
        end
        coefs, vifs = iridge(X, y, intercept, λ, cweights)

        mse, rmse, r2, adjr2 = iridge_stats(X, y, coefs, intercept, n, p, cweights)

        res_ridge_reg = ridgeRegRes(
            λ, p, n, intercept, coefs,
            vifs, mse, rmse, r2, adjr2, f, updatedformula, dataschema, isweighted, weights)

        return res_ridge_reg
    end

    function ridge(f::StatsModels.FormulaTerm, df::DataFrames.AbstractDataFrame, λs::AbstractRange ;
             weights::Union{Nothing,String}=nothing, remove_missing=false, contrasts=nothing )
            #  weights will be applied later
        X, y, n, p, intercept, f, copieddf, updatedformula, isweighted, dataschema = 
        design_matrix!(f, df, weights=weights, remove_missing=remove_missing, contrasts=contrasts, ridge=true)
   
        vcoefs, vvifs = iridge(X, y, intercept, λs)
        cweights = nothing
        if !isnothing(weights)
            cweights = copieddf[!, weights]
        end
        vcoefs, vvifs = iridge(X, y, intercept, λs, cweights)

        vmse = Vector{Float64}(undef, length(λs))
        vrmse = Vector{Float64}(undef, length(λs))
        vr2 = Vector{Float64}(undef, length(λs))
        vadjr2 = Vector{Float64}(undef, length(λs))

        for (i, x) in enumerate(λs)
            vmse[i], vrmse[i], vr2[i], vadjr2[i] = iridge_stats(X, y, vcoefs[i], intercept, n, p, cweights)
        end

        coefs_names = encapsulate_string(string.(StatsBase.coefnames(updatedformula.rhs)))
        vifs_names = "vif_" .* coefs_names
        cv = [λs vmse vrmse vr2 vadjr2 transpose(hcat(vcoefs...)) transpose(hcat(vvifs...)) ]
        all_names = ["λ", "MSE", "RMSE", "R2", "ADJR2", coefs_names..., vifs_names... ]
        df = DataFrame(cv, all_names)

        return df 
    end

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

function iridge(X_orig, y_orig, intercept, λ::Float64, weights::Union{Nothing,Vector{Float64}}=nothing)
    XTX, D, Z, ZTZ, ymean, Xmeans, y, X = prepare_ridge(X_orig, y_orig, intercept, weights)

    invZTZ = pinv(ZTZ + λ * I)
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

function iridge(X_orig, y_orig, intercept, λs::AbstractRange, weights::Union{Nothing,Vector{Float64}}=nothing)
    XTX, D, Z, ZTZ, ymean, Xmeans, y, X = prepare_ridge(X_orig, y_orig, intercept, weights)

    vcoefs = Vector{Vector{}}(undef, length(λs))
    vvifs = Vector{Vector{}}(undef, length(λs))

    for (i, λ) in enumerate(λs)
        invZTZ = pinv(ZTZ + λ * I)
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

struct ridgeRegRes
    λ::Float64
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
    println(io, "Lambda (λ):\t", rr.λ)
    println(io, "Model definition:\t", rr.modelformula)
    println(io, "Used observations:\t", rr.observations)
    println(io, "Model statistics:")
    @printf(io, "  R²: %g\t\t\tAdjusted R²: %g\n", rr.R2, rr.ADJR2)
    @printf(io, "  MSE: %g\t\t\tRMSE: %g\n", rr.MSE, rr.RMSE)

    helper_print_table(io, "Coefficients statistics:", 
        [rr.coefs, rr.VIF], ["Coefs", "VIF"], rr.updatedformula)

end
