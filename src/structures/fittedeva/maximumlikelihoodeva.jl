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
    transform(fm::MaximumLikelihoodEVA{BlockMaxima})::fittedEVA{BlockMaxima}

Transform the fitted model for the original covariate scales.
"""
function transform(fm::MaximumLikelihoodEVA{BlockMaxima{GeneralizedExtremeValue}})

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
