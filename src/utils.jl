"""
    pwm(x::Vector{<:Real}, p::Int,r::Int,s::Int)

Compute the empirical probability weighted moments defined by:
```math
M_{p,r,s} = \\mathbb{E}\\left[ X^p F^r(X) \\left\\{ 1-F(X) \\right\\}^s  \\right].
```
The unbiased empirical estimate is computed using the formula given by Landwehr et al. (1979).

*Reference:*
Landwehr, J. M., Matalas, N. C. and Wallis, J. R. (1979). Probability weighted moments compared with
    some traditional techniques in estimating Gumbel Parameters and quantiles. Water Resources Research,
    15(5), 1055–1064.
"""
function pwm(x::Vector{<:Real},p::Int,r::Int,s::Int)

    @assert sign(p)>=0 "p should be a non-negative integer."
    @assert sign(r)>=0 "r should be a non-negative integer."
    @assert sign(s)>=0 "s should be a non-negative integer."

    y = sort(x)
    n = length(y)

    m = 1/n*sum( y[i]^p * binomial(i-1,r)/binomial(n-1,r) * binomial(n-i,s)/binomial(n-1,s) for i=1:n )

    return m

end

"""
    slicematrix(A::AbstractMatrix{T}; dims::Int=1) where T

Convert a Matrix in a Vector of Vector. The slicing dimension can be defined with `dims`.
"""
function slicematrix(A::AbstractMatrix{T}; dims::Int=1) where T

    @assert dims∈(1,2) "dims should be either 1 or 2."

    n, m = size(A)

    if dims==1
        B = Vector{T}[Vector{T}(undef, n) for _ in 1:m]
        for i in 1:m
            B[i] .= A[:, i]
        end
    else
        B = Vector{T}[Vector{T}(undef, m) for _ in 1:n]
        for i in 1:n
            B[i] .= A[i, :]
        end
    end

    return B

end

"""
    unslicematrix(B::Array{Array{T,1},1}) where T

Convert a Vector of Vector in a Matrix. The vectors within each vector should be of the same length.
"""
function unslicematrix(B::Array{Array{T,1},1}; dims::Int=1) where T

    @assert dims∈(1,2) "dims should be either 1 or 2."

    m = length.(B)

    @assert all( ==(m[1]), m) "all vectors must have the same length."

    n = length(B)

    m = m[1]

    if dims == 1

        A = Matrix{T}(undef, m, n)

        for i in 1:n
            A[:,i] = B[i]
        end

    else

        A = Matrix{T}(undef, n, m)

        for i in 1:n
            A[i,:] = B[i]
        end

    end


    return A

end
