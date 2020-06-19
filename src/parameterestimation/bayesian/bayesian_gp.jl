"""
    gpfitbayes(y::Vector{<:Real};
         logscalecov::Vector{<:DataItem} = Vector{Variable}(),
         shapecov::Vector{<:DataItem} = Vector{Variable}(),
         niter::Int=5000, warmup::Int=2000)::BayesianEVA

Fit a non-stationary Generalized Pareto (GEV) distribution under the Bayesian paradigm to the vector of data contained in the Vector y.

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
ϕ = x*1/500
σ = exp.(ϕ)

# Sample from the non-stationary GEV distribution
pd = GeneralizedPareto.(σ,.1)
y = rand.(pd)

# Estimate the parameters
gpfitbayes(y, logscalecov = [Variable("x", x)])
```

"""
function gpfitbayes(y::Vector{<:Real};
     logscalecov::Vector{<:DataItem} = Vector{Variable}(),
     shapecov::Vector{<:DataItem} = Vector{Variable}(),
     niter::Int=5000, warmup::Int=2000)::BayesianEVA

     logscalecovstd = standardize.(logscalecov)
     shapecovstd = standardize.(shapecov)

    model = ThresholdExceedance(y, logscalecov = logscalecovstd, shapecov = shapecovstd)

    fittedmodel = fitbayes(model, niter=niter, warmup=warmup)

    return transform(fittedmodel)

end

"""
    gpfitbayes(df::DataFrame, datacol::Symbol;
        logscalecovid::Vector{Symbol}=Symbol[],
        shapecovid::Vector{Symbol}=Symbol[],
        niter::Int=5000, warmup::Int=2000)::MaximumLikelihoodEVA

Fit a Generalized Pareto (GP) distribution under the Bayesian paradigm to the vector of data contained in the dataframe `df` at the column `datacol`.

"""
function gpfitbayes(df::DataFrame, datacol::Symbol;
    logscalecovid::Vector{Symbol}=Symbol[],
    shapecovid::Vector{Symbol}=Symbol[],
    niter::Int=5000, warmup::Int=2000)::BayesianEVA

    logscalecov = buildVariables(df, logscalecovid)
    shapecov = buildVariables(df, shapecovid)

    fm = gpfitbayes(df[:,datacol], logscalecov = logscalecov, shapecov = shapecov, niter = niter, warmup = warmup)

    return fm

end

"""
    gpfitbayes(model::ThresholdExceedance, niter::Int=5000, warmup::Int=2000)::BayesianEVA

Fit the Generalized Pareto (GP) distribution under the Bayesian paradigm to the ThresholdExceedance model.

A random sample from the posterior distribution is generated using the NUTS algortihm.

Only flat prior is now supported.

"""
function gpfitbayes(model::ThresholdExceedance; niter::Int=5000, warmup::Int=2000)::BayesianEVA

    return fitbayes(model, niter=niter, warmup=warmup)

end
