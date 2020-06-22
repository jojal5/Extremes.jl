"""
    getinitialvalue(::Type{GeneralizedExtremeValue},y::Vector{<:Real})::Vector{<:Real}

Compute the initial values of the GEV parameters given the data `y`.

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
        σ₀ = scale(fd)
        ξ₀ = Distributions.shape(fd)
    else
        fm = gumbelfitpwm(y)
        μ₀ = fm.θ̂[1]
        σ₀ = fm.θ̂[2]
        ξ₀ = 0.0
    end

    initialvalues = [μ₀, log(σ₀), ξ₀]

    return initialvalues

end

"""
     getinitialvalue(::Type{GeneralizedPareto},y::Vector{<:Real})::Vector{<:Real}

Compute the initial values of the GP parameters given the data `y`.

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

Convert a Matrix in a Vector of Vector. The slicing dimension can be defined with `dims`.

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

Convert a Vector of Vector in a Matrix. The vectors within each vector should be of the same length.

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

Creates the explanatory variables with names corresponding to the symbols.

"""
function buildVariables(df::DataFrame, ids::Vector{Symbol})::Vector{Variable}

    variables = Vector{Variable}()

    for id in ids
        push!(variables, Variable(string(id), df[:,id]))
    end

    return variables

end
