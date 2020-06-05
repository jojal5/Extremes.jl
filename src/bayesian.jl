"""
    gevfitbayes(y::Vector{<:Real};
        locationcov::Vector{Vector{T}} where T<:Real = Vector{Vector{Float64}}(),
        scalecov::Vector{Vector{T}} where T<:Real = Vector{Vector{Float64}}(),
        shapecov::Vector{Vector{T}} where T<:Real = Vector{Vector{Float64}}(),
        niter::Int=5000, warmup::Int=2000)

Fit a non-stationary Generalized Extreme Value (GEV) distribution under the Bayesian paradigm to the vector of data contained in the vector y.

The optional parameter `locationcov` is a vector containing the covariates for the parameter μ.
The optional parameter `scalecov` is a vector containing the covariates for the parameter σ.
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
gevfitbayes(y, locationcov = [x])
```

"""
function gevfitbayes(y::Vector{<:Real};
    locationcov::Vector{ExplanatoryVariable} = Vector{ExplanatoryVariable}(),
    scalecov::Vector{ExplanatoryVariable} = Vector{ExplanatoryVariable}(),
    shapecov::Vector{ExplanatoryVariable} = Vector{ExplanatoryVariable}(),
    niter::Int=5000, warmup::Int=2000)::BayesianEVA

    model = BlockMaxima(y, locationcov = locationcov, scalecov = scalecov, shapecov = shapecov)

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

"""
    gpfitbayes(y::Vector{<:Real}, nobservation::Int; niter::Int=5000, warmup::Int=2000,
        threshold::Vector{<:Real}=[0], nobsperblock::Int=1, scalecov::Vector{Vector{T}} where T<:Real = Vector{Vector{Float64}}(),
        shapecov::Vector{Vector{T}} where T<:Real = Vector{Vector{Float64}}())::BayesianEVA

Fit a non-stationary Generalized Pareto (GEV) distribution under the Bayesian paradigm to the vector of data contained in the Vector y.

The optional parameter `scalecov` is a vector containing the covariates for the parameter σ.
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
gpfitbayes(y, scalecov = [x])
```
"""
function gpfitbayes(y::Vector{<:Real}, nobservation::Int; niter::Int=5000, warmup::Int=2000,
     threshold::Vector{<:Real}=[0], nobsperblock::Int=1, scalecov::Vector{ExplanatoryVariable} = Vector{ExplanatoryVariable}(),
     shapecov::Vector{ExplanatoryVariable} = Vector{ExplanatoryVariable}())::BayesianEVA

    model = PeaksOverThreshold(y, nobservation, threshold = threshold, nobsperblock = nobsperblock, scalecov = scalecov, shapecov = shapecov)

    fittedmodel = fitbayes(model, niter=niter, warmup=warmup)

    return fittedmodel

end

"""
    gpfitbayes(model::PeaksOverThreshold, niter::Int=5000, warmup::Int=2000)::BayesianEVA

Fit the Generalized Pareto (GP) distribution under the Bayesian paradigm to the PeaksOverThreshold model.

A random sample from the posterior distribution is generated using the NUTS algortihm.

Only flat prior is now supported.

"""
function gpfitbayes(model::PeaksOverThreshold, niter::Int=5000, warmup::Int=2000)::BayesianEVA

    return fitbayes(model, niter=niter, warmup=warmup)

end

"""
    fitbayes(model::EVA; niter::Int=5000, warmup::Int=2000)

Fits the extreme-value model under the Bayesian paradigm.
"""
function fitbayes(model::EVA; niter::Int=5000, warmup::Int=2000)::BayesianEVA

    # Set initial values to the maximum likelihood estimates
    ml = fit(model)
    initialvalues = ml.θ̂

    # Define the loglikelihood function and the gradient for the NUTS algorithm
    logf(θ::DenseVector) = loglike(model,θ)
    Δlogf(θ::DenseVector) = ForwardDiff.gradient(logf, θ)
    function logfgrad(θ::DenseVector)
        ll = logf(θ)
        g = Δlogf(θ)
        return ll, g
    end

    sim = Chains(niter, nparameter(model), start = (warmup + 1))
    θ = NUTSVariate(initialvalues, logfgrad)
    @showprogress for i in 1:niter
        sample!(θ, adapt = (i <= warmup))
        if i > warmup
            sim[i, :, 1] = θ
        end
    end

    fittedmodel = BayesianEVA(model, sim)

    return fittedmodel

end
