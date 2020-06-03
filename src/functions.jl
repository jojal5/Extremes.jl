
"""
Establish the parameter as function of the corresponding covariates.
"""
function computeparamfunction(data::Dict, covariateid::Array{Symbol,1}, n::Int) # TODO : Remove when no longer in use
    fun =
    if isempty(covariateid)
        function(β::Vector{<:Real})
            return identity(β)
        end
    else
        X = ones(n)

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
Establish the parameter as function of the corresponding covariates.
"""
function computeparamfunction(covariates::Vector{Vector{T}} where T<:Real)
    fun =
    if isempty(covariates)
        function(β::Vector{<:Real})
            return identity(β)
        end
    else
        X = ones(length(covariates[1]))

        for cov in covariates
            X = hcat(X, cov)
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

    # Fit the model with by the probability weigthed moments
    fm = Extremes.gevfitpwm(y)

    # Convert to fitted model in a Distribution object
    fd = Extremes.getdistribution(fm.model, fm.θ̂)[]

    # check if initial values are in the domain of the GEV
    valid_initialvalues = all(insupport(fd,y))

    #= If one at least one value does not lie in the support, then the initial
     values are replaced by the Gumbel initial values. =#
    if valid_initialvalues
        μ₀ = location(fd)
        σ₀ = scale(fd)
        ξ₀ = Distributions.shape(fd)
    else
        fm = Extremes.gumbelfitpwm(y)
        μ₀ = fm.θ̂[1]
        σ₀ = fm.θ̂[2]
        ξ₀ = 0.0
    end

    initialvalues = [μ₀, σ₀, ξ₀]

    return initialvalues

end

function getinitialvalue(::Type{GeneralizedPareto},y::Vector{<:Real})

    # Fit the model with by the probability weigthed moments
    fm = Extremes.gpfitpwm(y::Array{Float64})

    # Convert to fitted model in a Distribution object
    fd = Extremes.getdistribution(fm.model, fm.θ̂)[]

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
Get an initial values vector for the parameters of model
"""
function getinitialvalue(model::BlockMaxima)

    y = data(model)

    # Compute stationary initial values
    μ₀,σ₀,ξ₀ = Extremes.getinitialvalue(GeneralizedExtremeValue,y)
    # Store them in a dictionary
    θ₀ = Dict(:μ => μ₀, :ϕ => log(σ₀), :ξ => ξ₀)

    initialvalues = zeros(nparameter(model))
    pi = paramindex(model)
    for param in [:μ, :ϕ, :ξ]
        ind = pi[param][1]
        initialvalues[ind] = θ₀[param]
    end

    return initialvalues

end

"""
Get an initial values vector for the parameters of model
"""
function getinitialvalue(model::PeaksOverThreshold)

    y = model.data[model.dataid]

    # Compute stationary initial values
    σ₀,ξ₀ = Extremes.getinitialvalue(GeneralizedPareto,y)
    # Store them in a dictionary
    θ₀ = Dict(:ϕ => log(σ₀), :ξ => ξ₀)

    initialvalues = zeros(nparameter(model))
    pi = paramindex(model)
    for param in [:ϕ, :ξ]
        ind = pi[param][1]
        initialvalues[ind] = θ₀[param]
    end

    return initialvalues

end

"""
    getdistribution(model::EVA, θ::Vector{<:Real})

Return the fitted distribution in case of stationarity or the vector of fitted distribution in case of non-stationarity.
"""
function getdistribution(model::BlockMaxima, θ::Vector{<:Real})

    @assert length(θ)==nparameter(model) "The length of the parameter vector should be equal to the model number of parameters."

    pi = paramindex(model)
    μ = model.location.fun(θ[pi[:μ]])
    ϕ = model.logscale.fun(θ[pi[:ϕ]])
    ξ = model.shape.fun(θ[pi[:ξ]])

    σ = exp.(ϕ)

    fd = GeneralizedExtremeValue.(μ, σ, ξ)

    return fd

    # if length(fd) == 1
    #     res = fd[1]
    # else
    #     res = fd
    # end
    #
    # return res

end

"""
    getdistribution(model::EVA, θ::Vector{<:Real})

Return the fitted distribution in case of stationarity or the vector of fitted distribution in case of non-stationarity.
"""
function getdistribution(model::PeaksOverThreshold, θ::Vector{<:Real})

    @assert length(θ)==nparameter(model) "The length of the parameter vector should be equal to the model number of parameters."

    pi = paramindex(model)
    ϕ = model.logscalefun(θ[pi[:ϕ]])
    ξ = model.shapefun(θ[pi[:ξ]])

    σ = exp.(ϕ)

    fd = GeneralizedPareto.(σ, ξ)

    return fd

    # if length(fd) == 1
    #     res = fd[1]
    # else
    #     res = fd
    # end
    #
    # return res

end

"""
    getdistribution(fittedmodel::MaximumLikelihoodEVA)

Return the fitted distribution in case of stationarity or the vector of fitted distribution in case of non-stationarity.
"""
function getdistribution(fittedmodel::MaximumLikelihoodEVA)

    model = fittedmodel.model
    θ̂ = fittedmodel.θ̂

    res = getdistribution(model, θ̂)

    return res

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
Return the number of covariates.
"""
function getcovariatenumber(Covariate::Dict, params::Vector{Symbol}) # TODO : Remove when no longer in use

    ncovariate = 0

    for p in params
        if haskey(Covariate, p)
            ncovariate += length(Covariate[p])
        end
    end

    return ncovariate

end

"""
Return the number of covariates.
"""
function getcovariatenumber(model::BlockMaxima)
    return sum([length(model.location.covariate), length(model.logscale.covariate), length(model.shape.covariate)])
end


"""
Compute the model loglikelihood evaluated at θ.
"""
function loglike(model::EVA, θ::Vector{<:Real})

    y = data(model)

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
function paramindexing(Covariate::Dict, params::Vector{Symbol}) # TODO : Remove when not in use

    id = Symbol[]
    for p in params
        if haskey(Covariate,p)
            append!(id, fill(p, 1+length(Covariate[p])))
        else
            push!(id, p)
        end
    end

    paramindex = Dict{Symbol,Vector{<:Int}}()

    for p in params
        paramindex[p] = findall(id.==p)
    end

    return paramindex

end


"""
    quantile(model::EVA, θ::Vector{<:Real}, p::Real)

Compute the quantile of level `p` from the model evaluated at `θ"". If the model is non-stationary, then the effective quantiles are returned.
"""
function quantile(model::EVA, θ::Vector{<:Real}, p::Real)

    @assert zero(p)<p<one(p) "the quantile level should be between 0 and 1."

    pd = getdistribution(model, θ)

    q = quantile.(pd, p)

    return q

end

"""
Compute the quantile of level `p` from the fitted model by maximum likelihood. In the case of non-stationarity, the effective quantiles are returned.
"""
function quantile(fm::MaximumLikelihoodEVA, p::Real)

    @assert zero(p)<p<one(p) "the quantile level should be between 0 and 1."

    q = quantile(fm.model, fm.θ̂, p)

    return q

end

"""
    quantile(fm::Extremes.BayesianEVA,p::Real)

Compute the quantile of level `p` from the fitted Bayesian model `fm`. If the
model is stationary, then a quantile is returned for each MCMC steps. If the
model is non-stationary, a matrix of quantiles is returned, where each row
corresponds to a MCMC step and each column to a covariate.
"""
function quantile(fm::Extremes.BayesianEVA,p::Real)

    @assert zero(p)<p<one(p) "the quantile level should be between 0 and 1."

    θ = slicematrix(fm.sim.value[:,:,1], dims=2)

    q = quantile.(fm.model, θ, p)

    if !(typeof(q) <: Vector{<:Real})
        q = unslicematrix(q, dims=2)
    end

    return q

end


"""
    quantilevar(fd::Extremes.MaximumLikelihoodEVA, level::Real)

Compute the variance of the quantile of level `level` from the fitted model `fm`.
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

    return V

    # if isa(q, Real)
    #     res = V[1]
    # else
    #     res = V
    # end
    #
    # return res

end


function returnlevel(fm::MaximumLikelihoodEVA, returnPeriod::Real, confidencelevel::Real=.95)

      @assert returnPeriod > zero(returnPeriod) "the return period should be positive."
      @assert zero(confidencelevel)<confidencelevel<one(confidencelevel) "the confidence level should be in (0,1)."

      α = (1 - confidencelevel)

      # quantile level
      p = 1-1/returnPeriod

      q = Extremes.quantile(fm, p)

      v = Extremes.quantilevar(fm,p)

      qdist = Normal.(q,sqrt.(v))

      a = quantile.(qdist,α/2)
      b = quantile.(qdist,1-α/2)

      cint = Extremes.slicematrix(hcat(a,b), dims=2)

      res = ReturnLevel(fm, returnPeriod, q, cint)

      return res

end

function returnlevel(fm::BayesianEVA, returnPeriod::Real, confidencelevel::Real=.95)

      @assert returnPeriod > zero(returnPeriod) "the return period should be positive."
      @assert zero(confidencelevel)<confidencelevel<one(confidencelevel) "the confidence level should be in (0,1)."

      α = (1 - confidencelevel)

      # quantile level
      p = 1-1/returnPeriod

      Q = Extremes.quantile(fm, p)

      q = vec(mean(Q, dims=1))

      qsliced = Extremes.slicematrix(Q)

      a = quantile.(qsliced, α/2)
      b = quantile.(qsliced, 1-α/2)

      cint = Extremes.slicematrix(hcat(a,b), dims=2)

      res = ReturnLevel(fm, returnPeriod, q, cint)

      return res

end

"""
Get the number of parameters in a BlockMaxima
"""
function nparameter(model::BlockMaxima)
    return 3 + getcovariatenumber(model)
end

"""
Get the number of parameters in a PeaksOverThreshold
"""
function nparameter(model::PeaksOverThreshold)
    return 2 + getcovariatenumber(model.covariate, [:ϕ, :ξ])
end

"""
Get the parameter indexing for a BlockMaxima
"""
function paramindex(model::BlockMaxima)
    i = 0
    function increasei()
        i = i + 1
        return i
    end
    return Dict{Symbol,Vector{<:Int}}(
        :μ => Int64[increasei() for k in 1:length(model.location.covariate) + 1],
        :ϕ => Int64[increasei() for k in 1:length(model.logscale.covariate) + 1],
        :ξ => Int64[increasei() for k in 1:length(model.shape.covariate) + 1]
    )
end

"""
Get the parameter indexing for a PeaksOverThreshold
"""
function paramindex(model::PeaksOverThreshold)
    return paramindexing(model.covariate, [:ϕ, :ξ])
end

"""
Get the data for a BlockMaxima
"""
function data(model::BlockMaxima)
    return model.data
end

"""
Get the data for a PeaksOverThreshold
"""
function data(pot::PeaksOverThreshold)
    return pot.data[pot.dataid]
end

function Base.show(io::IO, obj::BlockMaxima)
  println(io, "Extreme value model")
  println(io, "    "*showparamfun(obj.location))
  println(io, "    "*showparamfun(obj.logscale))
  println(io, "    "*showparamfun(obj.shape))
end

function Base.show(io::IO, obj::PeaksOverThreshold)
  println(io, "Extreme value model")
  println(io, "    "*showparamfun(obj,:ϕ))
  println(io, "    "*showparamfun(obj,:ξ))
end

function Base.show(io::IO, obj::MaximumLikelihoodEVA)
    show(io, obj.model)
    println(io, "Maximum likelihood estimates")
    println(io, "θ̂ = $(obj.θ̂)")
end

function showparamfun(param::paramfun)
    covariate = [" + x$i" for i in 1:length(param.covariate)]
    res = string("$param ~ 1", covariate...)

    return res
end

function showparamfun(model::PeaksOverThreshold, param::Symbol) # TODO : Remove when not in use

    covariate = [" + $x" for x in model.covariate[param]]
    res = string("$param ~ 1", covariate...)

    return res

end
