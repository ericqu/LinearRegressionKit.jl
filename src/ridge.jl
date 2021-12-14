# # Ridge regression adapted from the SAS fomulas   
# # see https://blogs.sas.com/content/iml/2013/03/20/compute-ridge-regression.html

function iridge(X_orig, y_orig, intercept, λ::Float64)
    X = deepcopy(X_orig)
    y = deepcopy(y_orig)

    # removes the intercept (assumed to be the first column)
    if intercept
        X = X[:, deleteat!(collect(axes(X, 2)), 1)]
    end

    # get the means the Xs and ys
    Xmeans = mean(X, dims=1)
    ymean = mean(y)

    # center the X and y
    for i in 1:size(X, 2)
        X[:, i] .-= Xmeans[i]
    end
    y .-= ymean

    XTX = X'X
    D = Diagonal(diag(XTX))
    Z = X / sqrt(D)
    ZTZ = Z'Z 

    invZTZ = pinv(ZTZ + λ*I)
    coefs = invZTZ * (Z'*y) ./ (sqrt.(diag(XTX)))
    vifs = diag(invZTZ * ZTZ * invZTZ)
    
    if (intercept)
        # get intercept back
        interceptvalue = ymean - sum(vec(Xmeans) .* coefs)
        coefs = vec([interceptvalue coefs...])
        vifs = vec([0. vifs...])
    end

    return coefs, vifs
end

function iridge(X_orig, y_orig, intercept, λs::AbstractRange)
    X = deepcopy(X_orig)
    y = deepcopy(y_orig)

    # removes the intercept (assumed to be the first column)
    if intercept
        X = X[:, deleteat!(collect(axes(X, 2)), 1)]
    end

    # get the means the Xs and ys
    Xmeans = mean(X, dims=1)
    ymean = mean(y)

    # center the X and y
    for i in 1:size(X, 2)
        X[:, i] .-= Xmeans[i]
    end
    y .-= ymean

    XTX = X'X
    D = Diagonal(diag(XTX))
    Z = X / sqrt(D)
    ZTZ = Z'Z 

    vcoefs = Vector{Vector{}}(undef, length(λs))
    vvifs = Vector{Vector{}}(undef, length(λs))

    for (i, λ) in enumerate(λs)
        invZTZ = pinv(ZTZ + λ*I)
        vcoefs[i] = invZTZ * (Z'*y) ./ (sqrt.(diag(XTX)))
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
    ŷ= lr_predict(X, coefs, intercept)
    residuals = y .- ŷ
    sse = sum(residuals.^2)
    mse = sse / (n - p)
    rmse = real_sqrt(mse)

    sst = nothing
    if isnothing(weights)
        sst = getSST(y, intercept)
    else
        sst = getSST(y, intercept, weights)
    end
    r2 = 1. - (sse / sst)

    return mse, rmse, r2
end

"""
    function ridge(f::StatsModels.FormulaTerm, df::DataFrames.AbstractDataFrame, λ::Float64 ; 
    weights::Union{Nothing,String}=nothing, remove_missing=false, contrasts=nothing )

    Ridge regression, expects a λ parameter (also known as k).
"""
function ridge(f::StatsModels.FormulaTerm, df::DataFrames.AbstractDataFrame, λ::Float64 ; 
            weights::Union{Nothing,String}=nothing, remove_missing=false, contrasts=nothing )
    X, y, n, p, intercept, f, copieddf, updatedformula, isweighted, dataschema = 
        design_matrix!(f, df, weights=weights, remove_missing=remove_missing, contrasts=contrasts)

    coefs, vifs = iridge(X, y, intercept, λ)
    cweights = nothing
    if !isnothing(weights)
        cweights = copieddf[!, weights]
    end
    mse, rmse, r2 = iridge_stats(X, y, coefs, intercept, n, p, cweights)

    return coefs, mse, rmse, r2, vifs

end


function ridge(f::StatsModels.FormulaTerm, df::DataFrames.AbstractDataFrame, λs::AbstractRange ;
             weights::Union{Nothing,String}=nothing, remove_missing=false, contrasts=nothing )
    X, y, n, p, intercept, f, copieddf, updatedformula, isweighted, dataschema = 
    design_matrix!(f, df, weights=weights, remove_missing=remove_missing, contrasts=contrasts)
   
    vcoefs, vvifs = iridge(X, y, intercept, λs)
    cweights = nothing
    if !isnothing(weights)
        cweights = copieddf[!, weights]
    end

    vmse = Vector{Float64}(undef, length(λs))
    vrmse = Vector{Float64}(undef, length(λs))
    vr2 = Vector{Float64}(undef, length(λs))

    for (i, x) in enumerate(λs)
        vmse[i], vrmse[i], vr2[i] = iridge_stats(X, y, vcoefs[i], intercept, n, p, cweights)
    end
    return vcoefs, vmse, vrmse, vr2, vvifs
end