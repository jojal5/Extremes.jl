
"""
    delta(g::Function, θ̂::AbstractVector{<:Real}, H::AbstractPDMat)

Compute the variance of the function `g` of estimated paramters `θ̂` with negative observed information matrix `H`.

## Detail

`H`corresponds to the Hessian matrix of the negative log likelihood.
"""
function delta(g::Function, θ̂::AbstractVector{<:Real}, H::AbstractPDMat)
    
    ∇ = ForwardDiff.gradient(g, θ̂)
    
    v = invquad(H, ∇)
    
    return v
    
end


"""
    getinitialvalue(::Type{GeneralizedExtremeValue},y::Vector{<:Real})::Vector{<:Real}

Compute the initial values of the GEV parameters given the data `y`.

If the probability weighted moment estimations are valid, then those values are
returned. Otherwise, the probability weighted moment estimations of the Gumbel
distribution are returned.

# Example
```julia-repl
julia> Extremes.getinitialvalue(GeneralizedExtremeValue, y)
```
"""
function getinitialvalue(::Type{GeneralizedExtremeValue},y::Vector{<:Real})::Vector{<:Real}

    # Fit the model with by the probability weigthed moments
    fm = gevfitpwm(y)

    # Convert to fitted model in a Distribution object
    fd = getdistribution(fm.model, fm.θ̂)[]

    # check if initial values are in the domain of the GEV
    valid_initialvalues = all(insupport(fd,y))

    #= If one at least one value does not lie in the support, then the initial
     values are replaced by the Gumbel initial values. =#
    if valid_initialvalues
        μ₀ = location(fd)
        ϕ₀ = log(scale(fd))
        ξ₀ = Distributions.shape(fd)
    else
        fm = gumbelfitpwm(y)
        μ₀ = fm.θ̂[1]
        ϕ₀ = fm.θ̂[2]
        ξ₀ = 0.0
    end

    initialvalues = [μ₀, ϕ₀, ξ₀]

    return initialvalues

end

"""
    getinitialvalue(::Type{GeneralizedPareto},y::Vector{<:Real})::Vector{<:Real}

Compute the initial values of the GP distribution parameters given the data `y`.

If the probability weighted moment estimations are valid, then those values are
returned. Otherwise, the moment estimation of the exponential distribution is returned.

# Example
```julia-repl
julia> Extremes.getinitialvalue(GeneralizedPareto, y)
```
"""
function getinitialvalue(::Type{GeneralizedPareto},y::Vector{<:Real})::Vector{<:Real}

    # Fit the model with by the probability weigthed moments
    fm = gpfitpwm(y)

    # Convert to fitted model in a Distribution object
    fd = getdistribution(fm.model, fm.θ̂)[]

    if all(insupport(fd,y))
        σ₀ = scale(fd)
        ξ₀ = Distributions.shape(fd)
    else
        σ₀ = mean(y)
        ξ₀ = 0.0
    end

    initialvalues = [log(σ₀), ξ₀]

    return initialvalues

end


"""
    slicematrix(A::AbstractMatrix{T}; dims::Int=1)::Array{Array{T,1},1} where T

Convert a Matrix in a Vector of Vectors.

The slicing dimension is defined with the optional `dims` keyword.

# Examples
```julia-repl
julia> A = rand(1:10,5,4)
julia> V₁ = Extremes.slicematrix(A)
julia> V₂ = Extremes.slicematrix(A, dims=2)
```
"""
function slicematrix(A::AbstractMatrix{T}; dims::Int=1)::Array{Array{T,1},1} where T

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
    unslicematrix(B::Array{Array{T,1},1}; dims::Int=1)::AbstractMatrix{T} where T

Convert a Vector of Vectors in a Matrix if all vectors share the same length.

The concatenation dimension is defined with the optional `dims` keyword.

# Examples
```julia-repl
julia> V = Extremes.slicematrix(rand(1:10, 5, 4))
julia> A₁ = Extremes.unslicematrix(V)
julia> A₂ = Extremes.unslicematrix(V, dims=2)
```
"""
function unslicematrix(B::Array{Array{T,1},1}; dims::Int=1)::AbstractMatrix{T} where T

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

"""
    buildVariables(df::DataFrame, ids::Vector{Symbol})::Vector{Variable}

Build the `Variable` type from the columns `ids` of the DataFrame `df`.

# Example
```julia-repl
julia> df = Extremes.dataset("fremantle")
julia> Extremes.buildVariables(df, [:Year, :SOI])
```
"""
function buildVariables(df::DataFrame, ids::Vector{Symbol})::Vector{Variable}

    variables = Vector{Variable}()

    for id in ids
        push!(variables, Variable(string(id), df[:,id]))
    end

    return variables

end

"""
    Flat()

Construct a Flat <: ContinuousUnivariateDistribution object representing an improper uniform distribution on the real line.
"""
struct Flat <: ContinuousUnivariateDistribution end

minimum(pd::Flat) = -Inf
maximum(pd::Flat) = Inf
insupport(pd::Flat, y::Real) = true

logpdf(pd::Flat, y::Real) = 0.0


"""
    shape(pd::Gumbel)

Return the Gumbel distribution shape parameter value, *i.e.* 0.
"""
function shape(pd::Gumbel)
    return 0.
end