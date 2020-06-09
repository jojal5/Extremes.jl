"""
    gevfit(y::Vector{<:Real};
        locationcov::Vector{Vector{T}} where T<:Real = Vector{Vector{Float64}}(),
        logscalecov::Vector{Vector{T}} where T<:Real = Vector{Vector{Float64}}(),
        shapecov::Vector{Vector{T}} where T<:Real = Vector{Vector{Float64}}())

Fit the Generalized Extreme Value (GEV) distribution by maximum likelihood to the vector of data `y`.

The optional parameter `locationcov` is a vector containing the covariates for the parameter μ.
The optional parameter `logscalecov` is a vector containing the covariates for the parameter σ.
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
gevfit(y, locationcov = [ExplanatoryVariable("x", x)])
```

The covariate may be standardized to facilitate the estimation.

"""
function gevfit(y::Vector{<:Real};
    locationcov::Vector{ExplanatoryVariable} = Vector{ExplanatoryVariable}(),
    logscalecov::Vector{ExplanatoryVariable} = Vector{ExplanatoryVariable}(),
    shapecov::Vector{ExplanatoryVariable} = Vector{ExplanatoryVariable}())::MaximumLikelihoodEVA

    model = BlockMaxima(y, locationcov = locationcov, logscalecov = logscalecov, shapecov = shapecov)

    fittedmodel = fit(model)

    return fittedmodel

end

"""
    gevfit(df::DataFrame, datacol::Symbol;
        locationcovid::Vector{Symbol}=Symbol[],
        logscalecovid::Vector{Symbol}=Symbol[],
        shapecovid::Vector{Symbol}=Symbol[])::MaximumLikelihoodEVA

Fit the Generalized Extreme Value (GEV) distribution by maximum likelihood to the vector of data contained in the dataframe `df` at the column `datacol`.

"""
function gevfit(df::DataFrame, datacol::Symbol;
    locationcovid::Vector{Symbol}=Symbol[],
    logscalecovid::Vector{Symbol}=Symbol[],
    shapecovid::Vector{Symbol}=Symbol[])::MaximumLikelihoodEVA

    locationcov = buildExplanatoryVariables(df, locationcovid)
    logscalecov = buildExplanatoryVariables(df, logscalecovid)
    shapecov = buildExplanatoryVariables(df, shapecovid)

    model = BlockMaxima(df[:,datacol], locationcov = locationcov, logscalecov = logscalecov, shapecov = shapecov)

    fittedmodel = Extremes.fit(model)

    return fittedmodel

end


"""
    gevfit(model::BlockMaxima)

Fit the Generalized Extreme Value (GEV) distribution by maximum likelihood of the BlockMaxima model `model`.

"""
function gevfit(model::BlockMaxima)::MaximumLikelihoodEVA

    fit(model)

end

"""
    gpfit(y::Vector{<:Real})

Fit the Generalized Pareto (GP) distribution by maximum likelihood to the vector of data `y`.

The optional parameter `logscalecov` is a vector containing the covariates for the parameter σ.
The optional parameter `shapecov` is a vector containing the covariates for the parameter ξ.

Example with a non-stationary location parameter:
```julia
# Sample size
n = 300

# Covariate
x = collect(1:n)

# Location as function of the covariate
ϕ = x*1/500
σ = exp.(ϕ)

# Sample from the non-stationary GEV distribution
pd = GeneralizedPareto.(σ,.1)
y = rand.(pd)

# Estimate the parameters
gpfit(y, logscalecov = [ExplanatoryVariable("x", x)])
```

The covariate may be standardized to facilitate the estimation.

"""
function gpfit(y::Vector{<:Real};
    logscalecov::Vector{ExplanatoryVariable} = Vector{ExplanatoryVariable}(),
    shapecov::Vector{ExplanatoryVariable} = Vector{ExplanatoryVariable}())::MaximumLikelihoodEVA

    model = ThresholdExceedance(y, logscalecov = logscalecov, shapecov = shapecov)

    fittedmodel = fit(model)

    return fittedmodel

end

"""
    gpfit(df::DataFrame, datacol::Symbol;
        logscalecovid::Vector{Symbol}=Symbol[],
        shapecovid::Vector{Symbol}=Symbol[])::MaximumLikelihoodEVA

Fit a Generalized Pareto (GP) distribution by maximum likelihood to the vector of data contained in the dataframe `df` at the column `datacol`.

"""
function gpfit(df::DataFrame, datacol::Symbol;
    logscalecovid::Vector{Symbol}=Symbol[],
    shapecovid::Vector{Symbol}=Symbol[])::MaximumLikelihoodEVA

    logscalecov = buildExplanatoryVariables(df, logscalecovid)
    shapecov = buildExplanatoryVariables(df, shapecovid)

    model = ThresholdExceedance(df[:,datacol], logscalecov = logscalecov, shapecov = shapecov)

    fittedmodel = Extremes.fit(model)

    return fittedmodel

end

"""
    gpfit(model::ThresholdExceedance)::MaximumLikelihoodEVA

Fit the Generalized Pareto (GP) distribution by maximum likelihood to the ThresholdExceedance model.

"""
function gpfit(model::ThresholdExceedance)::MaximumLikelihoodEVA

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
