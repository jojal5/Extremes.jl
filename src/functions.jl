"""
    computeparamfunction(covariates::Vector{ExplanatoryVariable})::Function

Establish the parameter as function of the corresponding covariates.

"""
function computeparamfunction(covariates::Vector{ExplanatoryVariable})::Function

    fun =
    if isempty(covariates)
        function(β::Vector{<:Real})
            return identity(β)
        end
    else
        X = ones(length(covariates[1].value))

        for cov in covariates
            X = hcat(X, cov.value)
        end
        function(β::Vector{<:Real})
            return X*β
        end
    end
    return fun

end

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

"""
     getinitialvalue(::Type{GeneralizedPareto},y::Vector{<:Real})::Vector{<:Real}

Compute the initial values of the GP parameters given the data `y`.

"""
function getinitialvalue(::Type{GeneralizedPareto},y::Vector{<:Real})::Vector{<:Real}

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
    getinitialvalue(model::BlockMaxima)::Vector{<:Real}

Get an initial values vector for the parameters of model.

"""
function getinitialvalue(model::BlockMaxima)::Vector{<:Real}

    y = model.data

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
    getinitialvalue(model::ThresholdExceedance)::Vector{<:Real}

Get an initial values vector for the parameters of model.

"""
function getinitialvalue(model::ThresholdExceedance)::Vector{<:Real}

    y = model.data

    # Compute stationary initial values
    σ₀,ξ₀ = Extremes.getinitialvalue(GeneralizedPareto, y)
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
    getdistribution(model::BlockMaxima, θ::Vector{<:Real})::Vector{<:Distribution}

Return the fitted distribution in case of stationarity or the vector of fitted distribution in case of non-stationarity.

"""
function getdistribution(model::BlockMaxima, θ::Vector{<:Real})::Vector{<:Distribution}

    @assert length(θ)==nparameter(model) "The length of the parameter vector should be equal to the model number of parameters."

    pi = paramindex(model)
    μ = model.location.fun(θ[pi[:μ]])
    ϕ = model.logscale.fun(θ[pi[:ϕ]])
    ξ = model.shape.fun(θ[pi[:ξ]])

    σ = exp.(ϕ)

    fd = GeneralizedExtremeValue.(μ, σ, ξ)

    return fd

end

"""
    getdistribution(model::ThresholdExceedance, θ::Vector{<:Real})::Vector{<:Distribution}

Return the fitted distribution in case of stationarity or the vector of fitted distribution in case of non-stationarity.

"""
function getdistribution(model::ThresholdExceedance, θ::Vector{<:Real})::Vector{<:Distribution}

    @assert length(θ)==nparameter(model) "The length of the parameter vector should be equal to the model number of parameters."

    pi = paramindex(model)
    ϕ = model.logscale.fun(θ[pi[:ϕ]])
    ξ = model.shape.fun(θ[pi[:ξ]])

    σ = exp.(ϕ)

    fd = GeneralizedPareto.(σ, ξ)

    return fd

end

"""
    getdistribution(fittedmodel::MaximumLikelihoodEVA)::Vector{<:Distribution}

Return the fitted distribution in case of stationarity or the vector of fitted distribution in case of non-stationarity.

"""
function getdistribution(fittedmodel::MaximumLikelihoodEVA)::Vector{<:Distribution}

    model = fittedmodel.model
    θ̂ = fittedmodel.θ̂

    res = getdistribution(model, θ̂)

    return res

end

"""
    getcovariatenumber(model::ThresholdExceedance)::Int

Return the number of covariates.

"""
function getcovariatenumber(model::ThresholdExceedance)::Int

    return sum([length(model.logscale.covariate), length(model.shape.covariate)])

end

"""
    getcovariatenumber(model::BlockMaxima)::Int

Return the number of covariates.

"""
function getcovariatenumber(model::BlockMaxima)::Int

    return sum([length(model.location.covariate), length(model.logscale.covariate), length(model.shape.covariate)])

end


"""
    loglike(model::EVA, θ::Vector{<:Real})::Real

Compute the model loglikelihood evaluated at θ.

"""
function loglike(model::EVA, θ::Vector{<:Real})::Real

    y = model.data

    pd = getdistribution(model, θ)

    ll = sum(logpdf.(pd, y))

    return ll

end

"""
    loglike(fd::MaximumLikelihoodEVA)::Real

Compute the model loglikelihood evaluated at θ̂ if the maximum likelihood method has been used.

"""
function loglike(fd::MaximumLikelihoodEVA)::Real

    θ̂ = fd.results

    ll = loglike(fd.model, θ̂)

    return ll

end

"""
    parametervar(fm::Extremes.MaximumLikelihoodEVA)::Array{Float64, 2}

Compute the covariance parameters estimate of the fitted model `fm`.

"""
function parametervar(fm::MaximumLikelihoodEVA)::Array{Float64, 2}

    # Compute the parameters covariance matrix
    V = inv(hessian(fm))

    return V
end

"""
    quantile(model::EVA, θ::Vector{<:Real}, p::Real):Vector{<:Real}

Compute the quantile of level `p` from the model evaluated at `θ"". If the model is non-stationary, then the effective quantiles are returned.

"""
function quantile(model::EVA, θ::Vector{<:Real}, p::Real)::Vector{<:Real}

    @assert zero(p)<p<one(p) "the quantile level should be between 0 and 1."

    pd = getdistribution(model, θ)

    q = quantile.(pd, p)

    return q

end

"""
    quantile(fm::MaximumLikelihoodEVA, p::Real)::Vector{<:Real}

Compute the quantile of level `p` from the fitted model by maximum likelihood. In the case of non-stationarity, the effective quantiles are returned.

"""
function quantile(fm::MaximumLikelihoodEVA, p::Real)::Vector{<:Real}

    @assert zero(p)<p<one(p) "the quantile level should be between 0 and 1."

    q = quantile(fm.model, fm.θ̂, p)

    return q

end

"""
    quantile(fm::Extremes.BayesianEVA,p::Real)::Real

Compute the quantile of level `p` from the fitted Bayesian model `fm`. If the
model is stationary, then a quantile is returned for each MCMC steps. If the
model is non-stationary, a matrix of quantiles is returned, where each row
corresponds to a MCMC step and each column to a covariate.

"""
function quantile(fm::BayesianEVA,p::Real)::Real

    @assert zero(p)<p<one(p) "the quantile level should be between 0 and 1."

    θ = slicematrix(fm.sim.value[:,:,1], dims=2)

    q = quantile.(fm.model, θ, p)

    if !(typeof(q) <: Vector{<:Real})
        q = unslicematrix(q, dims=2)
    end

    return q

end


"""
    quantilevar(fd::Extremes.MaximumLikelihoodEVA, level::Real)::Vector{<:Real}

Compute the variance of the quantile of level `level` from the fitted model `fm`.

"""
function quantilevar(fm::MaximumLikelihoodEVA, level::Real)::Vector{<:Real}

    θ̂ = fm.θ̂
    H = hessian(fm)

    q = quantile(fm, level)

    V = zeros(length(q))

    for i=1:length(q)

        f(θ::DenseVector) = quantile(fm.model,θ,level)[i]
        Δf(θ::DenseVector) = ForwardDiff.gradient(f, θ)
        G = Δf(θ̂)

        V[i] = G'/H*G

    end

    return V

end

"""
    returnlevel(fm::MaximumLikelihoodEVA{BlockMaxima}, returnPeriod::Real, confidencelevel::Real=.95)::ReturnLevel

Compute the return level of the return period `returnPeriod` from the fitted model `fm`.

"""
function returnlevel(fm::MaximumLikelihoodEVA{BlockMaxima}, returnPeriod::Real, confidencelevel::Real=.95)::ReturnLevel

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

"""
    returnlevel(fm::MaximumLikelihoodEVA{ThresholdExceedance}, threshold::Vector{<:Real}, nobservation::Int,
        nobsperblock::Int, returnPeriod::Real, confidencelevel::Real=.95)::ReturnLevel

Compute the return level of the return period `returnPeriod` from the fitted model `fm`.

"""
function returnlevel(fm::MaximumLikelihoodEVA{ThresholdExceedance}, threshold::Vector{<:Real}, nobservation::Int,
    nobsperblock::Int, returnPeriod::Real, confidencelevel::Real=.95)::ReturnLevel

    # TODO : implement
    error("Not implemented")

end

"""
    returnlevel(fm::BayesianEVA{BlockMaxima}, returnPeriod::Real, confidencelevel::Real=.95)::ReturnLevel

Compute the return level of the return period `returnPeriod` from the fitted model `fm`.

"""
function returnlevel(fm::BayesianEVA{BlockMaxima}, returnPeriod::Real, confidencelevel::Real=.95)::ReturnLevel

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
    returnlevel(fm::BayesianEVA{ThresholdExceedance}, threshold::Vector{<:Real}, nobservation::Int,
        nobsperblock::Int, returnPeriod::Real, confidencelevel::Real=.95)::ReturnLevel

Compute the return level of the return period `returnPeriod` from the fitted model `fm`.

"""
function returnlevel(fm::BayesianEVA{ThresholdExceedance}, threshold::Vector{<:Real}, nobservation::Int,
    nobsperblock::Int, returnPeriod::Real, confidencelevel::Real=.95)::ReturnLevel

    # TODO : implement
    error("Not implemented")

end

"""
    nparameter(model::BlockMaxima)::Int

Get the number of parameters in a BlockMaxima.

"""
function nparameter(model::BlockMaxima)::Int

    return 3 + getcovariatenumber(model)

end

"""
    nparameter(model::ThresholdExceedance)::Int

Get the number of parameters in a ThresholdExceedance.

"""
function nparameter(model::ThresholdExceedance)::Int

    return 2 + getcovariatenumber(model)

end

"""
    paramindex(model::BlockMaxima)::Dict{Symbol,Vector{<:Int}}

Get the parameter indexing for a BlockMaxima.

"""
function paramindex(model::BlockMaxima)::Dict{Symbol,Vector{<:Int}}

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
    paramindex(model::ThresholdExceedance)::Dict{Symbol,Vector{<:Int}}

Get the parameter indexing for a ThresholdExceedance.

"""
function paramindex(model::ThresholdExceedance)::Dict{Symbol,Vector{<:Int}}

    i = 0
    function increasei()
        i = i + 1
        return i
    end
    return Dict{Symbol,Vector{<:Int}}(
        :ϕ => Int64[increasei() for k in 1:length(model.logscale.covariate) + 1],
        :ξ => Int64[increasei() for k in 1:length(model.shape.covariate) + 1]
    )

end

"""
    hessian(model::MaximumLikelihoodEVA)::Array{Float64, 2}

Calculates the Hessian matrix associated with the MaximumLikelihoodEVA model.
"""
function hessian(model::MaximumLikelihoodEVA)::Array{Float64, 2}

    fobj(θ) = -loglike(model.model, θ)
    return ForwardDiff.hessian(fobj, model.θ̂)

end

"""
    Base.show(io::IO, obj::EVA)

Override of the show function for the objects of type EVA.

"""
function Base.show(io::IO, obj::EVA)

    showEVA(io, obj)

end

"""
    showEVA(io::IO, obj::BlockMaxima; prefix::String = "")

Displays a BlockMaxima with the prefix `prefix` before every line.

"""
function showEVA(io::IO, obj::BlockMaxima; prefix::String = "")

    println(io, prefix, "BlockMaxima")
    println(io, prefix, "data :\t\t", typeof(obj.data), "[", length(obj.data), "]")
    println(io, prefix, "location :\t", showparamfun("μ", obj.location))
    println(io, prefix, "logscale :\t", showparamfun("ϕ", obj.logscale))
    println(io, prefix, "shape :\t\t", showparamfun("ξ", obj.shape))

end

"""
    showEVA(io::IO, obj::ThresholdExceedance; prefix::String = "")

Displays a ThresholdExceedance with the prefix `prefix` before every line.

"""
function showEVA(io::IO, obj::ThresholdExceedance; prefix::String = "")

    println(io, prefix, "ThresholdExceedance")
    println(io, prefix, "data :\t\t",typeof(obj.data), "[", length(obj.data), "]")
    println(io, prefix, "logscale :\t", showparamfun("ϕ", obj.logscale))
    println(io, prefix, "shape :\t\t", showparamfun("ξ", obj.shape))

end

"""
    Base.show(io::IO, obj::pwmEVA)

Override of the show function for the objects of type pwmEVA.

"""
function Base.show(io::IO, obj::pwmEVA)

    println(io, "pwmEVA")
    println("model :")
    showEVA(io, obj.model, prefix = "\t")
    println()
    println(io, "θ̂  :\t", obj.θ̂)

end

"""
    Base.show(io::IO, obj::MaximumLikelihoodEVA)

Override of the show function for the objects of type EVA.

"""
function Base.show(io::IO, obj::MaximumLikelihoodEVA)

    println(io, "MaximumLikelihoodEVA")
    println("model :")
    showEVA(io, obj.model, prefix = "\t")
    println()
    println(io, "θ̂  :\t", obj.θ̂)

end

"""
    Base.show(io::IO, obj::BayesianEVA)

Override of the show function for the objects of type EVA.

"""
function Base.show(io::IO, obj::BayesianEVA)

    println(io, "BayesianEVA")
    println("model :")
    showEVA(io, obj.model, prefix = "\t")
    println()
    println(io, "sim :\t", typeof(obj.sim))

end

"""
    showparamfun(name::String, param::paramfun)::String

Constructs a string describing a parameter `param` with name `name`.

"""
function showparamfun(name::String, param::paramfun)::String

    covariate = [" + $(x.name)" for x in param.covariate]
    res = string("$name ~ 1", covariate...)

    return res

end
