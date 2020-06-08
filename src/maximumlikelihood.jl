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
    locationcov::Vector{ExplanatoryVariable} = Vector{ExplanatoryVariable}(),
    scalecov::Vector{ExplanatoryVariable} = Vector{ExplanatoryVariable}(),
    shapecov::Vector{ExplanatoryVariable} = Vector{ExplanatoryVariable}())::MaximumLikelihoodEVA

    model = BlockMaxima(y, locationcov = locationcov, scalecov = scalecov, shapecov = shapecov)

    fittedmodel = fit(model)

    return fittedmodel

end

"""
    gevfit(df::DataFrame, datacol::Symbol; locationcovid::Vector{Symbol}=Symbol[], scalecovid::Vector{Symbol}=Symbol[], shapecovid::Vector{Symbol}=Symbol[])

Fit a non-stationary Generalized Extreme Value (GEV) distribution by maximum likelihood to the vector of data contained in the dataframe `df` at the column `datacol`.

"""
function gevfit(df::DataFrame, datacol::Symbol;
    locationcovid::Vector{Symbol}=Symbol[],
    scalecovid::Vector{Symbol}=Symbol[],
    shapecovid::Vector{Symbol}=Symbol[])::MaximumLikelihoodEVA

    locationcov = buildExplanatoryVariables(df, locationcovid)
    scalecov = buildExplanatoryVariables(df, scalecovid)
    shapecov = buildExplanatoryVariables(df, shapecovid)

    model = BlockMaxima(df[:,datacol], locationcov = locationcov, scalecov = scalecov, shapecov = shapecov)

    fittedmodel = Extremes.fit(model)

    return fittedmodel

end


"""
    gevfit(model::BlockMaxima)

Fit the non-stationary Generalized Extreme Value (GEV) distribution by maximum likelihood of the BlockMaxima model `model`.

"""
function gevfit(model::BlockMaxima)::MaximumLikelihoodEVA

    fit(model)

end

"""
    gpfit(y::Vector{<:Real})

Fit the Generalized Pareto (GP) distribution by maximum likelihood to the vector of data `y`.

"""
function gpfit(y::Vector{<:Real},
    nobservation::Int; threshold::Vector{<:Real}=[0], nobsperblock::Int=1,
    scalecov::Vector{ExplanatoryVariable} = Vector{ExplanatoryVariable}(),
    shapecov::Vector{ExplanatoryVariable} = Vector{ExplanatoryVariable}())::MaximumLikelihoodEVA

    model = PeaksOverThreshold(y, nobservation, threshold = threshold, nobsperblock = nobsperblock, scalecov = scalecov, shapecov = shapecov)

    fittedmodel = fit(model)

    return fittedmodel

end

"""
    gpfit(df::DataFrame, datacol::Symbol, nobservation::Int;
        threshold::Vector{<:Real}=[0], nobsperblock::Int=1,
        scalecovid::Vector{Symbol}=Symbol[],
        shapecovid::Vector{Symbol}=Symbol[])::MaximumLikelihoodEVA

Fit a Generalized Pareto (GP) distribution by maximum likelihood to the vector of data contained in the dataframe `df` at the column `datacol`.

"""
function gpfit(df::DataFrame, datacol::Symbol, nobservation::Int;
    threshold::Vector{<:Real}=[0], nobsperblock::Int=1,
    scalecovid::Vector{Symbol}=Symbol[],
    shapecovid::Vector{Symbol}=Symbol[])::MaximumLikelihoodEVA

    scalecov = buildExplanatoryVariables(df, scalecovid)
    shapecov = buildExplanatoryVariables(df, shapecovid)

    model = PeaksOverThreshold(df[:,datacol], nobservation, threshold = threshold, nobsperblock = nobsperblock, scalecov = scalecov, shapecov = shapecov)

    fittedmodel = Extremes.fit(model)

    return fittedmodel

end

"""
    gpfit(model::PeaksOverThreshold)::MaximumLikelihoodEVA

Fit the Generalized Pareto (GP) distribution by maximum likelihood to the PeaksOverThreshold model.

"""
function gpfit(model::PeaksOverThreshold)::MaximumLikelihoodEVA

    return fit(model)

end

"""
    fit(model::EVA)

Fit the extreme value model by maximum likelihood.

"""
function fit(model::EVA)::MaximumLikelihoodEVA

    initialvalues = getinitialvalue(model)

    fobj(θ) = -loglike(model, θ)

    res = optimize(fobj, initialvalues)

    if Optim.converged(res)
        θ̂ = Optim.minimizer(res)
    else
        @warn "The maximum likelihood algorithm did not find a solution. Maybe try with different initial values or with another method. The returned values are the initial values."
        θ̂ = initialvalues
    end

    fittedmodel = MaximumLikelihoodEVA(model, θ̂)

    return fittedmodel

end
