struct MaximumLikelihoodAbstractExtremeValueModel{T} <: AbstractFittedExtremeValueModel{T}
    "Extreme value model definition"
    model::T
    "Maximum likelihood estimate"
    θ̂::Vector{Float64}
end

"""
    aic(fm:::MaximumLikelihoodAbstractExtremeValueModel)

Compute the Akaike information criterion (AIC) of the fitted model by maximum likelihood method.

## Details

The AIC is defined as follows:

``AIC = 2 k - 2 \\log \\hat{L};``

where ``k`` is the number of estimated parameters and ``\\hat{L}`` is the maximized value of the likelihood function for the model. 



"""
function aic(fm::MaximumLikelihoodAbstractExtremeValueModel)
    return 2*nparameter(fm.model)-2*loglike(fm)
end

"""
    bic(fm:::MaximumLikelihoodAbstractExtremeValueModel)

Compute the Bayesian information criterion (BIC) of the fitted model by maximum likelihood method.

## Details

The BIC is defined as follows:

``BIC = k \\log n - 2 \\log \\hat{L};``

where ``k`` is the number of estimated parameters, ``n`` is the number of data and ``\\hat{L}`` is the maximized value of the likelihood function for the model. 

"""
function bic(fm::MaximumLikelihoodAbstractExtremeValueModel)
    n = length(fm.model.data.value)
    return nparameter(fm.model)*log(n)-2*loglike(fm)
end

"""
    hessian(model::MaximumLikelihoodAbstractExtremeValueModel)::PDMat{Float64, Matrix{Float64}}

Calculates the Hessian matrix associated with the MaximumLikelihoodAbstractExtremeValueModel model.
"""
function hessian(model::MaximumLikelihoodAbstractExtremeValueModel)::PDMat{Float64, Matrix{Float64}}

    fobj(θ) = -loglike(model.model, θ)
    H = ForwardDiff.hessian(fobj, model.θ̂)

    return PDMat(Symmetric(H))

end

"""
    parametervar(fm::MaximumLikelihoodAbstractExtremeValueModel)::Array{Float64, 2}

Compute the covariance parameters estimate of the fitted model `fm`.

"""
function parametervar(fm::MaximumLikelihoodAbstractExtremeValueModel)::Array{Float64, 2}

    # Compute the parameters covariance matrix
    V = inv(hessian(fm))

    return V
end

"""
    loglike(fd::MaximumLikelihoodAbstractExtremeValueModel)::Real

Compute the model loglikelihood AbstractExtremeValueModelluated at θ̂ if the maximum likelihood method has been used.

"""
function loglike(fm::MaximumLikelihoodAbstractExtremeValueModel)::Real

    ll = loglike(fm.model, fm.θ̂)

    return ll

end

"""
    getdistribution(fittedmodel::MaximumLikelihoodAbstractExtremeValueModel)::Vector{<:Distribution}

Return the fitted distribution in case of stationarity or the vector of fitted distribution in case of non-stationarity.

"""
function getdistribution(fittedmodel::MaximumLikelihoodAbstractExtremeValueModel)::Vector{<:Distribution}

    model = fittedmodel.model
    θ̂ = fittedmodel.θ̂

    res = getdistribution(model, θ̂)

    return res

end

"""
    quantile(fm::MaximumLikelihoodAbstractExtremeValueModel, p::Real)::Vector{<:Real}

Compute the quantile of level `p` from the fitted model by maximum likelihood. In the case of non-stationarity, the effective quantiles are returned.

"""
function quantile(fm::MaximumLikelihoodAbstractExtremeValueModel, p::Real)::Vector{<:Real}

    @assert zero(p)<p<one(p) "the quantile level should be between 0 and 1."

    q = quantile(fm.model, fm.θ̂, p)

    return q

end


"""
    quantilevar(fd::MaximumLikelihoodAbstractExtremeValueModel, level::Real)::Vector{<:Real}

Compute the variance of the quantile of level `level` from the fitted model `fm`.

"""
function quantilevar(fm::MaximumLikelihoodAbstractExtremeValueModel, level::Real)::Vector{<:Real}

    θ̂ = fm.θ̂
    H = Extremes.hessian(fm)

    q = quantile(fm, level)

    V = zeros(length(q))

    for i=1:length(q)

        g(θ::DenseVector) = Extremes.quantile(fm.model,θ,level)[i]
       
        V[i] = Extremes.delta(g, θ̂, H)

    end

    return V

end



"""
    returnlevel(fm::MaximumLikelihoodAbstractExtremeValueModel{BlockMaxima}, returnPeriod::Real)::ReturnLevel

Compute the return level corresponding to the return period `returnPeriod` from the fitted model `fm`.

"""
function returnlevel(fm::MaximumLikelihoodAbstractExtremeValueModel{BlockMaxima{T}}, returnPeriod::Real)::ReturnLevel where T

      @assert returnPeriod > zero(returnPeriod) "the return period should be positive."

      # quantile level
      p = 1-1/returnPeriod

      return ReturnLevel(BlockMaximaModel(fm), returnPeriod, quantile(fm, p))

end


function cint(rl::ReturnLevel{MaximumLikelihoodAbstractExtremeValueModel{BlockMaxima{T}}}, confidencelevel::Real=.95)::Vector{Vector{Real}} where T

      @assert rl.returnperiod > zero(rl.returnperiod) "the return period should be positive."
      @assert zero(confidencelevel)<confidencelevel<one(confidencelevel) "the confidence level should be in (0,1)."

      # quantile level
      p = 1-1/rl.returnperiod

      q = quantile(rl.model.fm, p)

      # Compute the credible interval

      α = (1 - confidencelevel)

      v = quantilevar(rl.model.fm,p)

      qdist = Normal.(q,sqrt.(v))

      a = quantile.(qdist,α/2)
      b = quantile.(qdist,1-α/2)

      return slicematrix(hcat(a,b), dims=2)

end

"""
    returnlevel(fm::MaximumLikelihoodAbstractExtremeValueModel{ThresholdExceedance}, threshold::Real, nobservation::Int,
        nobsperblock::Int, returnPeriod::Real)::ReturnLevel

Compute the return level corresponding to the return period `returnPeriod` from the fitted model `fm`.

The threshold should be a scalar. A varying threshold is not yet implemented.

"""
function returnlevel(fm::MaximumLikelihoodAbstractExtremeValueModel{ThresholdExceedance}, threshold::Real, nobservation::Int,
    nobsperblock::Int, returnPeriod::Real)::ReturnLevel

    @assert returnPeriod > zero(returnPeriod) "the return period should be positive."

    # Exceedance probability
    ζ = length(fm.model.data.value)/nobservation

    # Appropriate quantile level given the probability exceedance and the number of obs per year
    p = 1-1/(returnPeriod * nobsperblock * ζ)

    return ReturnLevel(PeakOverThreshold(fm, threshold, nobservation, nobsperblock),
        returnPeriod, threshold .+ quantile(fm, p))

end


function cint(rl::ReturnLevel{MaximumLikelihoodAbstractExtremeValueModel{ThresholdExceedance}}, confidencelevel::Real=.95)::Vector{Vector{Real}}

    @assert rl.returnperiod > zero(rl.returnperiod) "the return period should be positive."
    @assert zero(confidencelevel)<confidencelevel<one(confidencelevel) "the confidence level should be in (0,1)."

    # Exceedance probability
    ζ = length(rl.model.fm.model.data.value)/rl.model.nobservation

    # Appropriate quantile level given the probability exceedance and the number of obs per year
    p = 1-1/(rl.returnperiod * rl.model.nobsperblock * ζ)

    q = rl.model.threshold .+ quantile(rl.model.fm, p)

    # Compute the credible interval

    α = (1 - confidencelevel)

    # Computing the variance corresponding to ζ
    f(ζ::Real) = Extremes.quantile(rl.model.fm.model,rl.model.fm.θ̂,1-1/(rl.returnperiod * rl.model.nobsperblock * ζ))
    h = 1e-6 # Step for the finite difference approximation of the gradient
    Δf = (f(ζ + h) - f(ζ - h))/2/h
    v₁ = Δf.^2*ζ*(1-ζ)/rl.model.nobservation # Variance correponding to the estimates of ζ

    # This component seems to be forgoten by Coles (2001) in Section 4.4.1

    # Computing the variance corresponding to the other parameters
    v₂ = quantilevar(rl.model.fm, p)

    v = v₁ .+ v₂

    qdist = Normal.(q,sqrt.(v))

    a = quantile.(qdist,α/2)
    b = quantile.(qdist,1-α/2)

    return slicematrix(hcat(a,b), dims=2)

end

"""
    showAbstractFittedExtremeValueModel(io::IO, obj::MaximumLikelihoodAbstractExtremeValueModel; prefix::String = "")

Displays a MaximumLikelihoodAbstractExtremeValueModel with the prefix `prefix` before every line.

"""
function showAbstractFittedExtremeValueModel(io::IO, obj::MaximumLikelihoodAbstractExtremeValueModel; prefix::String = "")

    println(io, prefix, "MaximumLikelihoodAbstractExtremeValueModel")
    println(io, prefix, "model :")
    showAbstractExtremeValueModel(io, obj.model, prefix = prefix*"\t")
    println(io)
    println(io, prefix, "θ̂  :\t", obj.θ̂)

end

"""
    transform(fm::MaximumLikelihoodAbstractExtremeValueModel{BlockMaxima})::MaximumLikelihoodAbstractExtremeValueModel

Transform the fitted model for the original covariate scales.
"""
function transform(fm::MaximumLikelihoodAbstractExtremeValueModel{BlockMaxima{GeneralizedExtremeValue}})::MaximumLikelihoodAbstractExtremeValueModel

    locationcovstd = fm.model.location.covariate
    logscalecovstd = fm.model.logscale.covariate
    shapecovstd = fm.model.shape.covariate

    locationcov = Extremes.reconstruct.(locationcovstd)
    logscalecov = Extremes.reconstruct.(logscalecovstd)
    shapecov = Extremes.reconstruct.(shapecovstd)

    # Model on the original covariate scale
    model = BlockMaxima{GeneralizedExtremeValue}(fm.model.data, locationcov = locationcov, logscalecov = logscalecov, shapecov = shapecov)

    # Transformation of the parameter estimates
    θ̂ = deepcopy(fm.θ̂)
    ind = Extremes.paramindex(fm.model)

    for (var, par) in zip([locationcovstd, logscalecovstd, shapecovstd],[:μ, :ϕ, :ξ])
        if !isempty(var)
            a = getfield.(var, :scale)
            b = getfield.(var, :offset)

            # θ̂[ind[par][1]] = fm.θ̂[ind[par][1]] - sum( fm.θ̂[ind[par][1+i]] * b[i]/a[i] for i=1:length(a) )

            for i=1:length(a)
                θ̂[ind[par][1]] -= fm.θ̂[ind[par][1+i]]*b[i]/a[i]
                θ̂[ind[par][1+i]] = fm.θ̂[ind[par][1+i]]/a[i]
            end
        end
    end

    # Contruction of the AbstractFittedExtremeValueModel structure
    return MaximumLikelihoodAbstractExtremeValueModel(model, θ̂)

end

"""
    transform(fm::MaximumLikelihoodAbstractExtremeValueModel{BlockMaxima})::MaximumLikelihoodAbstractExtremeValueModel

Transform the fitted model for the original covariate scales.
"""
function transform(fm::MaximumLikelihoodAbstractExtremeValueModel{BlockMaxima{Gumbel}})::MaximumLikelihoodAbstractExtremeValueModel

    locationcovstd = fm.model.location.covariate
    logscalecovstd = fm.model.logscale.covariate

    locationcov = Extremes.reconstruct.(locationcovstd)
    logscalecov = Extremes.reconstruct.(logscalecovstd)

    # Model on the original covariate scale
    model = BlockMaxima{Gumbel}(fm.model.data, locationcov = locationcov, logscalecov = logscalecov)

    # Transformation of the parameter estimates
    θ̂ = deepcopy(fm.θ̂)
    ind = Extremes.paramindex(fm.model)

    for (var, par) in zip([locationcovstd, logscalecovstd],[:μ, :ϕ])
        if !isempty(var)
            a = getfield.(var, :scale)
            b = getfield.(var, :offset)

            for i=1:length(a)
                θ̂[ind[par][1]] -= fm.θ̂[ind[par][1+i]]*b[i]/a[i]
                θ̂[ind[par][1+i]] = fm.θ̂[ind[par][1+i]]/a[i]
            end
        end
    end

    # Contruction of the AbstractFittedExtremeValueModel structure
    return MaximumLikelihoodAbstractExtremeValueModel(model, θ̂)

end



"""
    transform(fm::MaximumLikelihoodAbstractExtremeValueModel{ThresholdExceedance})

Transform the fitted model for the original covariate scales.
"""
function transform(fm::MaximumLikelihoodAbstractExtremeValueModel{ThresholdExceedance})

    logscalecovstd = fm.model.logscale.covariate
    shapecovstd = fm.model.shape.covariate

    logscalecov = Extremes.reconstruct.(logscalecovstd)
    shapecov = Extremes.reconstruct.(shapecovstd)

    # Model on the original covariate scale
    model = ThresholdExceedance(fm.model.data, logscalecov = logscalecov, shapecov = shapecov)

    # Transformation of the parameter estimates
    θ̂ = deepcopy(fm.θ̂)
    ind = Extremes.paramindex(fm.model)

    for (var, par) in zip([logscalecovstd, shapecovstd],[:ϕ, :ξ])
        if !isempty(var)
            a = getfield.(var, :scale)
            b = getfield.(var, :offset)

            # θ̂[ind[par][1]] = fm.θ̂[ind[par][1]] - sum( fm.θ̂[ind[par][1+i]] * b[i]/a[i] for i=1:length(a) )

            for i=1:length(a)
                θ̂[ind[par][1]] -= fm.θ̂[ind[par][1+i]]*b[i]/a[i]
                θ̂[ind[par][1+i]] = fm.θ̂[ind[par][1+i]]/a[i]
            end
        end
    end

    # Contruction of the AbstractFittedExtremeValueModel structure
    return MaximumLikelihoodAbstractExtremeValueModel(model, θ̂)

end


function cint(fm::MaximumLikelihoodAbstractExtremeValueModel, clevel::Real=.95)::Array{Array{Float64,1},1}

    @assert 0<clevel<1 "the confidence level should be between 0 and 1."

    α = 1 - clevel

    V = parametervar(fm)

    confint = Vector{Vector{Float64}}()

    q = quantile.(Normal(0,1),[α/2, 1 - α/2])

    for i in eachindex(fm.θ̂)
        push!(confint, fm.θ̂[i] .+ q*sqrt(V[i,i]))
    end

    return confint

end
