"""
    slicematrix(A::AbstractMatrix{T}; dims::Int=1) where T

Convert a Matrix in a Vector of Vector. The slicing dimension can be defined with `dims`.
"""
function slicematrix(A::AbstractMatrix{T}; dims::Int=1) where T

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
function unslicematrix(B::Array{Array{T,1},1}) where T

    m = length.(B)

    @assert all( ==(m[1]), m) "all vectors must have the same length."

    n = length(B)

    m = m[1]

    A = Matrix{T}(undef, n, m)

    for i in 1:n
        A[i,:] .= B[i]
    end

    return A

end
