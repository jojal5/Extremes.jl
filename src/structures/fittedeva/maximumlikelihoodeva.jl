struct MaximumLikelihoodEVA{T} <: fittedEVA{T}
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
function loglike(fm::MaximumLikelihoodEVA)::Real

    ll = loglike(fm.model, fm.θ̂)

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
    returnlevel(fm::MaximumLikelihoodEVA{BlockMaxima}, returnPeriod::Real)::ReturnLevel

Compute the return level corresponding to the return period `returnPeriod` from the fitted model `fm`.

"""
function returnlevel(fm::MaximumLikelihoodEVA{BlockMaxima}, returnPeriod::Real)::ReturnLevel

      @assert returnPeriod > zero(returnPeriod) "the return period should be positive."

      # quantile level
      p = 1-1/returnPeriod

      return ReturnLevel(BlockMaximaModel(fm), returnPeriod, quantile(fm, p))

end

"""
    cint(rl::ReturnLevel{MaximumLikelihoodEVA{BlockMaxima}}, confidencelevel::Real=.95)::Vector{Vector{Real}}

Compute the confidence intervel for the return level corresponding to the return period
`returnPeriod` from the fitted model `fm` with confidence level `confidencelevel`.

"""
function cint(rl::ReturnLevel{MaximumLikelihoodEVA{BlockMaxima}}, confidencelevel::Real=.95)::Vector{Vector{Real}}

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
    returnlevel(fm::MaximumLikelihoodEVA{ThresholdExceedance}, threshold::Real, nobservation::Int,
        nobsperblock::Int, returnPeriod::Real)::ReturnLevel

Compute the return level corresponding to the return period `returnPeriod` from the fitted model `fm`.

The threshold should be a scalar. A varying threshold is not yet implemented.

"""
function returnlevel(fm::MaximumLikelihoodEVA{ThresholdExceedance}, threshold::Real, nobservation::Int,
    nobsperblock::Int, returnPeriod::Real)::ReturnLevel

    @assert returnPeriod > zero(returnPeriod) "the return period should be positive."

    # Exceedance probability
    ζ = length(fm.model.data.value)/nobservation

    # Appropriate quantile level given the probability exceedance and the number of obs per year
    p = 1-1/(returnPeriod * nobsperblock * ζ)

    return ReturnLevel(PeakOverThreshold(fm, threshold, nobservation, nobsperblock),
        returnPeriod, threshold .+ quantile(fm, p))

end


"""
    cint(rl::ReturnLevel{MaximumLikelihoodEVA{ThresholdExceedance}}, confidencelevel::Real=.95)::Vector{Vector{Real}}

Compute the confidence intervel for the return level corresponding to the return period
`returnPeriod` from the fitted model `fm` with confidence level `confidencelevel`.

The threshold should be a scalar. A varying threshold is not yet implemented.

"""
function cint(rl::ReturnLevel{MaximumLikelihoodEVA{ThresholdExceedance}}, confidencelevel::Real=.95)::Vector{Vector{Real}}

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
    f(θ::Vector{<:Real}) = Extremes.quantile(rl.model.fm.model,rl.model.fm.θ̂,1-1/(rl.returnperiod * rl.model.nobsperblock * θ[]))[]
    v₁ = (ForwardDiff.gradient(f, [ζ])[])^2*ζ*(1-ζ)/rl.model.nobservation

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
    showfittedEVA(io::IO, obj::MaximumLikelihoodEVA; prefix::String = "")

Displays a MaximumLikelihoodEVA with the prefix `prefix` before every line.

"""
function showfittedEVA(io::IO, obj::MaximumLikelihoodEVA; prefix::String = "")

    println(io, prefix, "MaximumLikelihoodEVA")
    println(io, prefix, "model :")
    showEVA(io, obj.model, prefix = prefix*"\t")
    println(io)
    println(io, prefix, "θ̂  :\t", obj.θ̂)

end

"""
    transform(fm::MaximumLikelihoodEVA{BlockMaxima})::MaximumLikelihoodEVA

Transform the fitted model for the original covariate scales.
"""
function transform(fm::MaximumLikelihoodEVA{BlockMaxima})::MaximumLikelihoodEVA

    locationcovstd = fm.model.location.covariate
    logscalecovstd = fm.model.logscale.covariate
    shapecovstd = fm.model.shape.covariate

    locationcov = Extremes.reconstruct.(locationcovstd)
    logscalecov = Extremes.reconstruct.(logscalecovstd)
    shapecov = Extremes.reconstruct.(shapecovstd)

    # Model on the original covariate scale
    model = BlockMaxima(fm.model.data, locationcov = locationcov, logscalecov = logscalecov, shapecov = shapecov)

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

    # Contruction of the fittedEVA structure
    return MaximumLikelihoodEVA(model, θ̂)

end



"""
    transform(fm::MaximumLikelihoodEVA{ThresholdExceedance})

Transform the fitted model for the original covariate scales.
"""
function transform(fm::MaximumLikelihoodEVA{ThresholdExceedance})

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

    # Contruction of the fittedEVA structure
    return MaximumLikelihoodEVA(model, θ̂)

end


"""
    cint(fm::MaximumLikelihoodEVA, clevel::Real=.95)::Array{Array{Float64,1},1}

Compute the Wald parameter confidence intervals using the approximate parameter estimates covariance matrix.
"""
function cint(fm::MaximumLikelihoodEVA, clevel::Real=.95)::Array{Array{Float64,1},1}

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
