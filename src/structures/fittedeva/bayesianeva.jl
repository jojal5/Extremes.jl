struct BayesianEVA{T<:EVA} <: fittedEVA
    "Extreme value model definition"
    model::T
    "MCMC outputs"
    sim::Mamba.Chains
end

"""
    quantile(fm::BayesianEVA,p::Real)::Real

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
    returnlevel(fm::BayesianEVA{BlockMaxima}, returnPeriod::Real, confidencelevel::Real=.95)::ReturnLevel

Compute the return level of the return period `returnPeriod` from the fitted model `fm`.

"""
function returnlevel(fm::BayesianEVA{BlockMaxima{GeneralizedExtremeValue}}, returnPeriod::Real, confidencelevel::Real=.95)::ReturnLevel

      @assert returnPeriod > zero(returnPeriod) "the return period should be positive."
      @assert zero(confidencelevel)<confidencelevel<one(confidencelevel) "the confidence level should be in (0,1)."

      α = (1 - confidencelevel)

      # quantile level
      p = 1-1/returnPeriod

      Q = quantile(fm, p)

      q = vec(mean(Q, dims=1))

      qsliced = slicematrix(Q)

      a = quantile.(qsliced, α/2)
      b = quantile.(qsliced, 1-α/2)

      cint = slicematrix(hcat(a,b), dims=2)

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
    Base.show(io::IO, obj::BayesianEVA)

Override of the show function for the objects of type BayesianEVA.

"""
function Base.show(io::IO, obj::BayesianEVA)

    println(io, "BayesianEVA")
    println(io, "model :")
    showEVA(io, obj.model, prefix = "\t")
    println(io)
    println(io, "sim :\t", typeof(obj.sim))

end


"""
    transform(fm::BayesianEVA{BlockMaxima{GeneralizedExtremeValue}})

Transform the fitted model for the original covariate scales.
"""
function transform(fm::BayesianEVA{BlockMaxima{GeneralizedExtremeValue}})

    locationcovstd = fm.model.location.covariate
    logscalecovstd = fm.model.logscale.covariate
    shapecovstd = fm.model.shape.covariate

    locationcov = Extremes.reconstruct.(locationcovstd)
    logscalecov = Extremes.reconstruct.(logscalecovstd)
    shapecov = Extremes.reconstruct.(shapecovstd)

    # Model on the original covariate scale
    model = BlockMaxima(fm.model.data, locationcov = locationcov, logscalecov = logscalecov, shapecov = shapecov)

    # Transformation of the parameter estimates
    z = fm.sim.value[:,:,1]

    x = deepcopy(z)
    ind = Extremes.paramindex(fm.model)

    for (var, par) in zip([locationcovstd, logscalecovstd, shapecovstd],[:μ, :ϕ, :ξ])
        if !isempty(var)
            a = getfield.(var, :scale)
            b = getfield.(var, :offset)

            for i=1:length(a)
                x[:,ind[par][1]] = x[:,ind[par][1]] - z[:,ind[par][1+i]] * b[i]/a[i]
                x[:,ind[par][1+i]] = z[:,ind[par][1+i]]/a[i]
            end
        end
    end

    sim = fm.sim
    sim.value[:,:,1] = x

    # Contruction of the fittedEVA structure
    return BayesianEVA(model, sim)

end



"""
    transform(fm::BayesianEVA{ThresholdExceedance})

Transform the fitted model for the original covariate scales.
"""
function transform(fm::BayesianEVA{ThresholdExceedance})

    logscalecovstd = fm.model.logscale.covariate
    shapecovstd = fm.model.shape.covariate

    logscalecov = Extremes.reconstruct.(logscalecovstd)
    shapecov = Extremes.reconstruct.(shapecovstd)

    # Model on the original covariate scale
    model = ThresholdExceedance(fm.model.data, logscalecov = logscalecov, shapecov = shapecov)

    # Transformation of the parameter estimates
    z = fm.sim.value[:,:,1]

    x = deepcopy(z)
    ind = Extremes.paramindex(fm.model)

    for (var, par) in zip([logscalecovstd, shapecovstd],[:ϕ, :ξ])
        if !isempty(var)
            a = getfield.(var, :scale)
            b = getfield.(var, :offset)

            for i=1:length(a)
                x[:,ind[par][1]] = x[:,ind[par][1]] - z[:,ind[par][1+i]] * b[i]/a[i]
                x[:,ind[par][1+i]] = z[:,ind[par][1+i]]/a[i]
            end
        end
    end

    sim = fm.sim
    sim.value[:,:,1] = x

    # Contruction of the fittedEVA structure
    return BayesianEVA(model, sim)

end
