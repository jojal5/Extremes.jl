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
