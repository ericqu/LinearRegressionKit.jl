using LinearAlgebra
import LinearAlgebra:checksquare

"""
    function sweep_linreg(x, y)

    Convenience function when the design matrix (the xs and ys) are known. And when only the coefficients are needed.
    Uses the normal equations and the sweep operator.
    similar to the x \\ y operator.
"""
function sweep_linreg(x, y)
    nn, p = size(x)
    if ndims(y) == 2
        n, pp = size(y)
    else 
        pp = 1
    end
    back = pp - 1

    xy = [x y]
    xytxy = xy' * xy
    sweep_op_full!(xytxy)

    return xytxy'[1:p, end-back:end]
end

"""
    function sweep_op_internal!(A::AbstractMatrix{Float64}, k::Int64)

    Implement the naive sweep operator described by Goodnight, J. (1979). "A Tutorial on the SWEEP Operator." The American Statistician.
    The algorithm is modified to work on column major to limit cache misses (although comment are referring to the original algorithm).
    It gives the transpose of the results.
"""
function sweep_op_internal!(A::AbstractMatrix{Float64}, k::Int64)
    p = checksquare(A)
    if k >= p 
        throw(ArgumentError("Incorrect k value"))
    end
    
    @inbounds D = A[k, k] # step 1 D = Aₖₖ
    
    if D == zero(eltype(A))
        throw(ArgumentError("sweep_op_internal!: the element $k,$k of the matrix is zero. Is the Design Matrix symmetric positive definite?"))
    end

    @inbounds for i in 1:p
        A[i, k] = A[i, k] / D   # step 2 divide row k by D
    end

    @inbounds for i in 1:(k - 1) # step 3: for every other row  i != k 
        B = A[k, i]   # let B = Aᵢₖ
        for j in 1:p
            A[j, i] = A[j, i] - B * A[j, k]  # Subtract B * row k from row i
        end
        A[k, i] = - B / D # set Aᵢₖ = -B/D    
    end 

    @inbounds for i in k + 1:p # step 3: for every other row  i != k 
        B = A[k, i]   # let B = Aᵢₖ
        for j in 1:p
            A[j, i] = A[j, i] - B * A[j, k]  # Subtract B * row k from row i
        end
        A[k, i] = - B / D # set Aᵢₖ = -B/D    
    end 

    @inbounds A[k,k] = 1 / D # step 4 Set Aₖₖ = 1/D
    
    return nothing
end

# function sweep_op!(A::AbstractMatrix{Float64}, k::Int64)
#     sweep_op_internal!(A, k)
# end

# function sweep_op!(A::AbstractMatrix{Float64}, ks::AbstractVector{Int64})
#     for k in ks
#         sweep_op_internal!(A, k)
#     end
#     return A
# end

"""
    function sweep_op_full!(A::AbstractMatrix{Float64})

    (internal) Get SSE, error sum of squares, for the full model.
"""
function sweep_op_full!(A::AbstractMatrix{Float64})
    n , p = size(A)

    for k in 1:p-1
        sweep_op_internal!(A, k)
    end
    return A[p,p]
end

"""
    function sweep_op_fullT1SS!(A::AbstractMatrix{Float64})

    (internal) Get SSE, error sum of squares for the full model.
    Also give Type I SS for all independent variables.
"""
function sweep_op_fullT1SS!(A::AbstractMatrix{Float64})
    n , p = size(A)
    TypeISS = Vector{Float64}(undef, p-1)

    for k in 1:p-1
        preSSE = A[p,p]
        sweep_op_internal!(A, k)
        TypeISS[k] = preSSE - A[p,p]
    end
    return A[p,p], TypeISS
end

"""
    function sweep_op_fullT1SS!(A::AbstractMatrix{Float64})

    (internal) Get SSE, error sum of squares for all (p-1) models.
    Also give Type I SS for all independent variables.
"""
function sweep_op_allT1SS!(A::AbstractMatrix{Float64})
    n , p = size(A)
    TypeISS = Vector{Float64}(undef, p-1)
    SSEs = Vector{Float64}(undef, p-1)

    for k in 1:p-1
        preSSE = A[p,p]
        sweep_op_internal!(A, k)
        SSEs[k] = A[p, p]
        TypeISS[k] = preSSE - A[p,p]
    end
    return SSEs, TypeISS
end

"""
    function get_TypeIISS(extended_inverse::AbstractMatrix{Float64})

    (internal) Get Type II SS for all independent variables given an extended inverse (sweep operator already applied)
"""
function get_TypeIISS(extended_inverse::AbstractMatrix{Float64})
    n, p = size(extended_inverse')
    TypeIISS = Vector{Float64}(undef, p-1)

    for k in 1:p-1
        TypeIISS[k] = extended_inverse'[k, end]^2 / extended_inverse'[k, k]
    end

    return TypeIISS
end