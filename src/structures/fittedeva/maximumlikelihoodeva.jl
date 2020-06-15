struct MaximumLikelihoodEVA{T<:EVA} <: fittedEVA
    "Extreme value model definition"
    model::T
    "Maximum likelihood estimate"
    θ̂::Vector{Float64}
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
    parametervar(fm::MaximumLikelihoodEVA)::Array{Float64, 2}

Compute the covariance parameters estimate of the fitted model `fm`.

"""
function parametervar(fm::MaximumLikelihoodEVA)::Array{Float64, 2}

    # Compute the parameters covariance matrix
    V = inv(hessian(fm))

    return V
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
    quantile(fm::MaximumLikelihoodEVA, p::Real)::Vector{<:Real}

Compute the quantile of level `p` from the fitted model by maximum likelihood. In the case of non-stationarity, the effective quantiles are returned.

"""
function quantile(fm::MaximumLikelihoodEVA, p::Real)::Vector{<:Real}

    @assert zero(p)<p<one(p) "the quantile level should be between 0 and 1."

    q = quantile(fm.model, fm.θ̂, p)

    return q

end

"""
    quantilevar(fd::MaximumLikelihoodEVA, level::Real)::Vector{<:Real}

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

      q = quantile(fm, p)

      v = quantilevar(fm,p)

      qdist = Normal.(q,sqrt.(v))

      a = quantile.(qdist,α/2)
      b = quantile.(qdist,1-α/2)

      cint = slicematrix(hcat(a,b), dims=2)

      res = ReturnLevel(fm, returnPeriod, q, cint)

      return res

end

"""
    returnlevel(fm::MaximumLikelihoodEVA{ThresholdExceedance}, threshold::Vector{<:Real}, nobservation::Int,
        nobsperblock::Int, returnPeriod::Real, confidencelevel::Real=.95)::ReturnLevel

Compute the return level of the return period `returnPeriod` from the fitted model `fm`.

The threshold should be a scalar. A varying threshold is not yet implemented.

"""
function returnlevel(fm::MaximumLikelihoodEVA{ThresholdExceedance}, threshold::Real, nobservation::Int,
    nobsperblock::Int, returnPeriod::Real, confidencelevel::Real=.95)::ReturnLevel

    @assert returnPeriod > zero(returnPeriod) "the return period should be positive."
    @assert zero(confidencelevel)<confidencelevel<one(confidencelevel) "the confidence level should be in (0,1)."

    α = (1 - confidencelevel)

    # Exceedance probability
    ζ = length(fm.model.data)/nobservation

    # Appropriate quantile level given the probability exceedance and the number of obs per year
    p = 1-1/(returnPeriod * nobsperblock * ζ)

    q = threshold .+ quantile(fm, p)

    # Computing the variance corresponding to ζ
    f(θ::Vector{<:Real}) = Extremes.quantile(fm.model,fm.θ̂,1-1/(returnPeriod * nobsperblock * θ[]))[]
    v₁ = (ForwardDiff.gradient(f, [ζ])[])^2*ζ*(1-ζ)/nobservation

    # This component seems to be forgoten by Coles (2001) in Section 4.4.1

    # Computing the variance corresponding to the other parameters
    v₂ = Extremes.quantilevar(fm, p)

    v = v₁ .+ v₂

    qdist = Normal.(q,sqrt.(v))

    a = quantile.(qdist,α/2)
    b = quantile.(qdist,1-α/2)

    cint = Extremes.slicematrix(hcat(a,b), dims=2)

    res = ReturnLevel(fm, returnPeriod, q, cint)

    return res

end

"""
    Base.show(io::IO, obj::MaximumLikelihoodEVA)

Override of the show function for the objects of type EVA.

"""
function Base.show(io::IO, obj::MaximumLikelihoodEVA)

    println(io, "MaximumLikelihoodEVA")
    println(io, "model :")
    showEVA(io, obj.model, prefix = "\t")
    println(io)
    println(io, "θ̂  :\t", obj.θ̂)

end
