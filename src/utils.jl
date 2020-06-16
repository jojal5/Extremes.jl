"""
    getcluster(y::Array{<:Real,1}, u₁::Real , u₂::Real=0)::DataFrame

Returns a DataFrame with clusters for exceedance models. A cluster is defined as a sequence where values are higher than u₂ with at least a value higher than threshold u₁.

"""
function getcluster(y::Array{<:Real,1}, u₁::Real , u₂::Real=0.0)::DataFrame

    n = length(y)

    clusterBegin  = Int64[]
    clusterLength = Int64[]
    clusterMax    = Float64[]
    clusterPosMax = Int64[]
    clusterSum    = Float64[]

    exceedancePosition = findall(y .> u₁)

    clusterEnd = 0

    for i in exceedancePosition

            if i > clusterEnd

               j = 1

                while (i-j) > 0
                    if y[i-j] > u₂
                        j += 1
                    else
                        break
                    end
                end

                k = 1

                while (i+k) < (n+1)
                    if y[i+k] > u₂
                        k += 1
                    else
                        break
                    end
                end

            ind = i-(j-1) : i+(k-1)

            maxval, idxmax = findmax(y[ind])

            push!(clusterMax, maxval)
            push!(clusterPosMax, idxmax+ind[1]-1)
            push!(clusterSum, sum(y[ind]) )
            push!(clusterLength, length(ind) )
            push!(clusterBegin, ind[1] )

            clusterEnd = ind[end]

            end

    end


    P = clusterMax./clusterSum

    cluster = DataFrame(Begin=clusterBegin, Length=clusterLength, Max=clusterMax, Position=clusterPosMax, Sum=clusterSum, P=P)

    return cluster

end

"""
    getcluster(df::DataFrame, u₁::Real, u₂::Real=0.0)::DataFrame

Returns a DataFrame with clusters for exceedance models. A cluster is defined as a sequence where values are higher than u₂ with at least a value higher than threshold u₁.

"""
function getcluster(df::DataFrame, u₁::Real, u₂::Real=0.0)::DataFrame

    coltype = describe(df)[:,:eltype]

    @assert coltype[1]==Date || coltype[1]==DateTime "The first dataframe column should be of type Date."
    @assert coltype[2]<:Real "The second dataframe column should be of any subtypes of Real."

    cluster = DataFrame(Begin=Int64[], Length=Int64[], Max=Float64[], Position=Int64[], Sum=Float64[], P=Float64[])

    years = unique(year.(df[:,1]))

    for yr in years

        ind = year.(df[:,1]) .== yr
        c = getcluster(df[ind,2], u₁, u₂)
        c[!,:Begin] = findfirst(ind) .+ c[:,:Begin] .- 1
        append!(cluster, c)

    end

    cluster[!,:Begin] = df[cluster[:,:Begin],1]

    return cluster

end

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
    buildExplanatoryVariables(df::DataFrame, ids::Vector{Symbol})::Vector{ExplanatoryVariable}

Creates the explanatory variables with names corresponding to the symbols.

"""
function buildExplanatoryVariables(df::DataFrame, ids::Vector{Symbol})::Vector{ExplanatoryVariable}

    variables = Vector{ExplanatoryVariable}()

    for id in ids
        push!(variables, ExplanatoryVariable(string(id), df[:,id]))
    end

    return variables

end
