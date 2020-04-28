"""
    GeneralizedExtremeValue()

Create a *standard* `GeneralizedExtremeValue` object.
"""
function GeneralizedExtremeValue()
    res = Distributions.GeneralizedExtremeValue(0,1,0)
end

"""
    getcluster(y::Array{<:Real,1}, u₁::Real , u₂::Real=0)

Returns a DataFrame with clusters for exceedance models. A cluster is defined as a sequence where values are higher than u₂ with at least a value higher than threshold u₁.
"""
function getcluster(y::Array{<:Real,1}, u₁::Real , u₂::Real=0.0)

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
    getcluster(df::DataFrame, u₁::Real, u₂::Real=0)

Returns a DataFrame with clusters for exceedance models. A cluster is defined as a sequence where values are higher than u₂ with at least a value higher than threshold u₁.
"""
function getcluster(df::DataFrame, u₁::Real, u₂::Real=0.0)

    coltype = describe(df)[:eltype]#colwise(eltype, df)

    @assert coltype[1]==Date || coltype[1]==DateTime "The first dataframe column should be of type Date."
    @assert coltype[2]<:Real "The second dataframe column should be of any subtypes of Real."

    cluster = DataFrame(Begin=Int64[], Length=Int64[], Max=Float64[], Position=Int64[], Sum=Float64[], P=Float64[])

    years = unique(year.(df[1]))

    for yr in years

        ind = year.(df[1]) .== yr
        c = getcluster(df[ind,2], u₁, u₂)
        c[:Begin] = findfirst(ind) .+ c[:Begin] .-1
        append!(cluster, c)

    end

    d = df[1]
    cluster[:Begin] = d[cluster[:Begin]]

    return cluster

end

"""
    getinitialvalue(dist::GeneralizedExtremeValue,y::Vector{<:Real})

Compute the initial values of the GEV parameters given the data `y`.

"""
function getinitialvalue(dist::Distributions.GeneralizedExtremeValue,y::Vector{<:Real})

    pd = Extremes.gevfitlmom(y)

    # check if initial values are in the domain of the GEV
    valid_initialvalues = all(insupport(pd,y))

    if valid_initialvalues
        μ₀ = location(pd)
        σ₀ = scale(pd)
        ξ₀ = Distributions.shape(pd)
    else
        pd = Extremes.gumbelfitpwmom(y)
        μ₀ = location(pd)
        σ₀ = scale(pd)
        ξ₀ = 0.0
    end

    initialvalues = [μ₀, σ₀, ξ₀]

    return initialvalues

end

"""
    getinitialvalue(dist::GeneralizedPareto,y::Vector{<:Real})

Compute the initial values of the GPD parameters given the data `y`.
The threshold is assumed to be 0.

"""
function getinitialvalue(dist::Distributions.GeneralizedPareto,y::Vector{<:Real})

    fd = Extremes.gpdfitmom(y::Array{Float64}, threshold=0.0)

    if all(insupport(fd,y))
        σ₀ = scale(fd)
        ξ₀ = Distributions.shape(fd)
    else
        σ₀ = mean(y)
        ξ₀ = 0.0
    end

    initialvalues = [σ₀, ξ₀]

    return initialvalues

end

"""
    gevfitlmom(x::Array{T,1} where T<:Real)

Fit a GEV distribution with L-Moment method.
"""
function gevfitlmom(x::Array{T,1} where T<:Real)

    n = length(x)
    y = sort(x)
    r = 1:n

    #     L-Moments estimations (Cunnane, 1989)
    b₀ = mean(y)
    b₁ = sum( (r .- 1).*y )/n/(n-1)
    b₂ = sum( (r .- 1).*(r .- 2).*y ) /n/(n-1)/(n-2)

    # GEV parameters estimations
    c = (2b₁ - b₀)/(3b₂ - b₀) - log(2)/log(3)
    k = 7.859c + 2.9554c^2
    σ̂ = k *( 2b₁-b₀ ) /(1-2^(-k))/gamma(1+k)
    μ̂ = b₀ - σ̂/k*( 1-gamma(1+k) )

    ξ̂ = -k

    pdfit = GeneralizedExtremeValue(μ̂,σ̂,ξ̂)

    return pdfit
end




# """
#     gevhessian(y::Array{N,1} where N<:Real,μ::Real,σ::Real,ξ::Real)
#
# Hessian matrix...
# """
# function gevhessian(y::Array{T,1} where T<:Real,μ::Real,σ::Real,ξ::Real)
#
#     #= Estimate the hessian matrix evaluated at (μ, σ, ξ) for the iid gev random sample y =#
#
#     logl(θ) = loglikelihood(GeneralizedExtremeValue(θ...),y)
#
#     H = ForwardDiff.hessian(logl, [μ σ ξ])
#
# end
#

"""
    gpdfitmom(y::Array{T} where T<:Real; threshold::Real=0.0)

Fit a Generalized Pareto Distribution over y.
"""
function gpdfitmom(y::Array{T} where T<:Real; threshold::Real=0.0)

    if isapprox(threshold,0)
        ȳ = mean(y)
        s² = var(y)
    else
        ȳ = mean(y .- threshold)
        s² = var(y .- threshold)
    end

    ξ̂ = 1/2*(1-ȳ^2/s²)
    σ̂ = (1-ξ̂)*ȳ

    return GeneralizedPareto(threshold,σ̂,ξ̂)

end

"""
    gumbelfitpwmom(x::Array{T,1} where T<:Real)

Fits a Gumbel distribution using ...
"""
function gumbelfitpwmom(x::Array{T,1} where T<:Real)

    n = length(x)
    y = sort(x)
    r = 1:n

    # Probability weighted moments
    b₀ = mean(y)
    b₁ = 1/n/(n-1)*sum( y[i]*(n-i) for i=1:n)

    # Gumbel parameters estimations
    σ̂ = (b₀ - 2*b₁)/log(2)
    μ̂ = b₀ - Base.MathConstants.eulergamma*σ̂

    pdfit = Gumbel(μ̂,σ̂)

    return pdfit
end
