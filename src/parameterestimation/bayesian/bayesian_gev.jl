"""
    gevfitbayes(y::Vector{<:Real};
        locationcov::Vector{Vector{T}} where T<:Real = Vector{Vector{Float64}}(),
        logscalecov::Vector{Vector{T}} where T<:Real = Vector{Vector{Float64}}(),
        shapecov::Vector{Vector{T}} where T<:Real = Vector{Vector{Float64}}(),
        niter::Int=5000, warmup::Int=2000)

Fit the Generalized Extreme Value (GEV) distribution under the Bayesian paradigm to the vector of data `y`.

The optional parameter `locationcov` is a vector containing the covariates for the parameter μ.
The optional parameter `logscalecov` is a vector containing the covariates for the parameter σ.
The optional parameter `shapecov` is a vector containing the covariates for the parameter ξ.

The covariate may be standardized to facilitate the estimation.

A random sample of the posterior distribution is generated using the NUTS algortihm.

Only flat prior is now supported.

Example with a non-stationary location parameter:
```julia
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
gevfitbayes(y, locationcov = [ExplanatoryVariable("x", x)])
```

"""
function gevfitbayes(y::Vector{<:Real};
    locationcov::Vector{ExplanatoryVariable} = Vector{ExplanatoryVariable}(),
    logscalecov::Vector{ExplanatoryVariable} = Vector{ExplanatoryVariable}(),
    shapecov::Vector{ExplanatoryVariable} = Vector{ExplanatoryVariable}(),
    niter::Int=5000, warmup::Int=2000)::BayesianEVA

    model = BlockMaxima(y, locationcov = locationcov, logscalecov = logscalecov, shapecov = shapecov)

    fittedmodel = fitbayes(model, niter=niter, warmup=warmup)

    return fittedmodel

end

"""
    gevfitbayes(df::DataFrame, datacol::Symbol;
        locationcovid::Vector{Symbol}=Symbol[],
        logscalecovid::Vector{Symbol}=Symbol[],
        shapecovid::Vector{Symbol}=Symbol[],
        niter::Int=5000, warmup::Int=2000)::MaximumLikelihoodEVA

Fit a Generalized Extreme Value (GEV) distribution under the Bayesian paradigm to the vector of data contained in the dataframe `df` at the column `datacol`.

"""
function gevfitbayes(df::DataFrame, datacol::Symbol;
    locationcovid::Vector{Symbol}=Symbol[],
    logscalecovid::Vector{Symbol}=Symbol[],
    shapecovid::Vector{Symbol}=Symbol[],
    niter::Int=5000, warmup::Int=2000)::BayesianEVA

    locationcov = buildExplanatoryVariables(df, locationcovid)
    logscalecov = buildExplanatoryVariables(df, logscalecovid)
    shapecov = buildExplanatoryVariables(df, shapecovid)

    model = BlockMaxima(df[:,datacol], locationcov = locationcov, logscalecov = logscalecov, shapecov = shapecov)

    fittedmodel = fitbayes(model, niter=niter, warmup=warmup)

    return fittedmodel

end

"""
    gevfitbayes(model::BlockMaxima; niter::Int=5000, warmup::Int=2000)

Fit a non-stationary Generalized Extreme Value (GEV) distribution under the Bayesian paradigm of the BlockMaxima model `model`.

"""
function gevfitbayes(model::BlockMaxima; niter::Int=5000, warmup::Int=2000)::BayesianEVA

    return fitbayes(model, niter=niter, warmup=warmup)

end
