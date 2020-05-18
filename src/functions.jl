
"""
Establish the parameter as function of the corresponding covariates.
"""
function computeparamfunction(data::Dict, covariateid::Array{Symbol,1})
    fun =
    if isempty(covariateid)
        function(β::Vector{<:Real})
            return identity(β)
        end
    else
        X = ones(data[:n])

        for i=1:length(covariateid)
            X = hcat(X, data[covariateid[i]])
        end
        function(β::Vector{<:Real})
            return X*β
        end
    end
    return fun
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

    coltype = describe(df)[:,:eltype]#colwise(eltype, df)

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
    getinitialvalue(dist::GeneralizedExtremeValue,y::Vector{<:Real})

Compute the initial values of the GEV parameters given the data `y`.

"""
function getinitialvalue(::Type{GeneralizedExtremeValue},y::Vector{<:Real})

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

function getinitialvalue(::Type{GeneralizedPareto},y::Vector{<:Real})

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
    getdistribution(model::EVA, θ::Vector{<:Real})

Return the fitted distribution in case of stationarity or the vector of fitted distribution in case of non-stationarity.
"""
function getdistribution(model::EVA, θ::Vector{<:Real})

    dist = model.distribution

    μ = model.locationfun(θ[model.paramindex[:μ]])
    ϕ = model.logscalefun(θ[model.paramindex[:ϕ]])
    ξ = model.shapefun(θ[model.paramindex[:ξ]])

    σ = exp.(ϕ)

    fd = dist.(μ, σ, ξ)

    if length(fd) == 1
        res = fd[1]
    else
        res = fd
    end

    return res

end

"""
Get an initial values vector for the parameters of model
"""
function getinitialvalue(model::EVA)

    dist = model.distribution
    y = model.data[model.dataid]

    # Compute stationary initial values
    μ₀,σ₀,ξ₀ = Extremes.getinitialvalue(dist,y)
    # Store them in a dictionary
    θ₀ = Dict(:μ => μ₀, :ϕ => log(σ₀), :ξ => ξ₀)

    initialvalues = zeros(model.nparameters)
    for param in [:μ, :ϕ, :ξ]
        ind = model.paramindex[param][1]
        initialvalues[ind] = θ₀[param]
    end

    return initialvalues

end

"""
Return the parameter number of the model
"""
function getparameternumber(Covariate::Dict)

    # The number of parameters in the model without any covariates
    nparameters = 3

    for p in [:μ, :ϕ, :ξ]
        if haskey(Covariate, p)
            nparameters += length(Covariate[p])
        end
    end

    return nparameters

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

"""
Compute the model loglikelihood evaluated at θ.
"""
function loglike(model::EVA, θ::Vector{<:Real})

    y = model.data[model.dataid]

    pd = getdistribution(model, θ)

    ll = sum(logpdf.(pd, y))

    return ll

end

"""
Compute the model loglikelihood evaluated at θ̂ if the maximum likelihood method has been used.
"""
function loglike(fd::MaximumLikelihoodEVA)

    θ̂ = fd.results

    ll = loglike(fd.model, θ̂)

    return ll

end

"""
    parametervar(fm::Extremes.MaximumLikelihoodEVA)

Compute the covariance parameters estimate of the fitted model `fm`.
"""
function parametervar(fm::Extremes.MaximumLikelihoodEVA)

    # Compute the parameters covariance matrix
    V = inv(fm.H)

    return V
end

"""
Return the indexes of parameters belonging to the locationFunction,
logscaleFunction and shapeFunction from a single vector.
"""
function paramindexing(Covariate::Dict)

    params = [:μ, :ϕ, :ξ]

    id = Symbol[]
    for p in params
        if haskey(Covariate,p)
            append!(id, fill(p, 1+length(Covariate[p])))
        else
            push!(id, p)
        end
    end

    paramindex = Dict(:μ => findall(id.==:μ), :ϕ => findall(id.==:ϕ), :ξ => findall(id.==:ξ))

    return paramindex

end

"""
Compute the quantile of level `p` from the fitted model for the data at index = `index, in case of non-stationarity
"""
function quantile(model::MaximumLikelihoodEVA, p::Real)

    @assert zero(p)<p<one(p) "the quantile level should be between 0 and 1."

    fd = getdistribution(model)

    r = quantile.(fd, p)

    return r

end

"""
Compute the quantile of level `p` from the fitted model. If the model is non-stationary, then the effective quantile are returned.
"""
function quantile(model::EVA, θ::Vector{<:Real}, p::Real)

    @assert zero(p)<p<one(p) "the quantile level should be between 0 and 1."

    pd = getdistribution(model, θ)

    r = quantile.(pd, p)

    return r

end


"""
    quantilevar(fd::Extremes.MaximumLikelihoodEVA, level::Real)

Compute the variance of quantile of level `level`from the fitted model `fd.
"""
function quantilevar(fm::Extremes.MaximumLikelihoodEVA, level::Real)

    θ̂ = fm.θ̂
    H = fm.H

    q = quantile(fm, level)

    V = zeros(length(q))

    for i=1:length(q)

        f(θ::DenseVector) = quantile(fm.model,θ,level)[i]
        Δf(θ::DenseVector) = ForwardDiff.gradient(f, θ)
        G = Δf(θ̂)

        V[i] = G'/H*G

    end

    if isa(q, Real)
        res = V[1]
    else
        res = V
    end

    return res

end


function Base.show(io::IO, obj::EVA)
  println(io, "Extreme value model")
  println(io, "Model: $(obj.distribution)")
  println(io, "    "*showparamfun(obj,:μ))
  println(io, "    "*showparamfun(obj,:ϕ))
  println(io, "    "*showparamfun(obj,:ξ))
end

function Base.show(io::IO, obj::MaximumLikelihoodEVA)
    show(io, obj.model)
    println(io, "Maximum likelihood estimates")
    println(io, "θ̂ = $(obj.θ̂)")
end

function showparamfun(model::Extremes.EVA,param::Symbol)

    covariate = [" + $x" for x in model.covariate[param]]
    res = string("$param ~ 1", covariate...)

    return res

end
