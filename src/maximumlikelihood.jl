"""
    gevfit(y::Vector{<:Real};
        locationcov::Vector{Vector{T}} where T<:Real = Vector{Vector{Float64}}(),
        scalecov::Vector{Vector{T}} where T<:Real = Vector{Vector{Float64}}(),
        shapecov::Vector{Vector{T}} where T<:Real = Vector{Vector{Float64}}())

Fit a non-stationary Generalized Extreme Value (GEV) distribution by maximum likelihood to the vector of data contained in `y`.

The optional parameter `locationcov` is a vector containing the covariates for the parameter μ.
The optional parameter `scalecov` is a vector containing the covariates for the parameter σ.
The optional parameter `shapecov` is a vector containing the covariates for the parameter ξ.

Example with a non-stationary location parameter:
```julia
using Extremes, Distributions

# Sample size
n = 300

# Covariate
x = collect(1:n)

# Location as function of the covariate
μ = x*1/100

# Sample from the non-stationary GEV distribution
pd = GeneralizedExtremeValue.(μ,1,.1)
y = rand.(pd)

# Estimate the parameters
gevfit(y, locationcov = [x])
```

The covariate may be standardized to facilitate the estimation.

"""
function gevfit(y::Vector{<:Real};
    locationcov::Vector{Vector{T}} where T<:Real = Vector{Vector{Float64}}(),
    scalecov::Vector{Vector{T}} where T<:Real = Vector{Vector{Float64}}(),
    shapecov::Vector{Vector{T}} where T<:Real = Vector{Vector{Float64}}())

    model = BlockMaxima(y, locationcov = locationcov, scalecov = scalecov, shapecov = shapecov)

    fittedmodel = fit(model)

    return fittedmodel

end

"""
    gevfit(model::BlockMaxima)

Fit the non-stationary Generalized Extreme Value (GEV) distribution by maximum likelihood of the BlockMaxima model `model`.

"""
function gevfit(model::BlockMaxima)

    fit(model)

end

"""
    gpfit(y::Vector{<:Real})

Fit the Generalized Extreme Value (GEV) distribution by maximum likelihood to the vector of data `y.
"""
function gpfit(y::Vector{<:Real}, nobservation::Int; threshold::Vector{<:Real}=[0], nobsperblock::Int=1)

    model = PeaksOverThreshold(y, nobservation, threshold = threshold, nobsperblock = nobsperblock)

    fittedmodel = fit(model)

    return fittedmodel

end


function gpfit(data::Dict, dataid::Symbol, nobservation::Int ; Covariate::Dict=Dict{Symbol,Vector{Symbol}}(), threshold::Vector{<:Real}=[0], nobsperblock::Int=1)

    # Put empty Symbol array to stationary parameters
    for k in [:ϕ, :ξ]
        if !(haskey(Covariate,k))
            Covariate[k] = Symbol[]
        end
    end

    model = PeaksOverThreshold(data[dataid], nobservation,
        scalecov = [data[s] for s in Covariate[:ϕ]],
        shapecov = [data[s] for s in Covariate[:ξ]],
        threshold = threshold, nobsperblock = nobsperblock)

    fittedmodel = fit(model)

    return fittedmodel

end

"""
    gpfit(model::EVA)

Fit the Generalized Pareto (GP) distribution by maximum likelihood to the EVA model.

"""
function gpfit(model::EVA)

    fit(model)

end


"""
    fit(model::EVA)

Fit the extreme value model by maximum likelihood.

"""
function fit(model::EVA)

    initialvalues = getinitialvalue(model)

    fobj(θ) = -loglike(model, θ)

    res = optimize(fobj, initialvalues)

    if Optim.converged(res)
        θ̂ = Optim.minimizer(res)
    else
        @warn "The maximum likelihood algorithm did not find a solution. Maybe try with different initial values or with another method. The returned values are the initial values."
        θ̂ = initialvalues
    end

    H = ForwardDiff.hessian(fobj, θ̂)

    fittedmodel = MaximumLikelihoodEVA(model, θ̂, H)

    return fittedmodel

end
