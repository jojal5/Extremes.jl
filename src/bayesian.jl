"""
    gevfitbayes(y::DenseArray; niter::Int=5000, warmup::Int=2000)

Fit the Generalized Extreme Value (GEV) distribution under the Bayesian paradigm to the vector of data `y`.

Data is a dictionary with Symbol as keys.

A random sample from the posterior distribution is generated using the NUTS algortihm.

Only flat prior is now supported.
"""
function gevfitbayes(y::Vector{<:Real}; niter::Int=5000, warmup::Int=2000)

    model = BlockMaxima(y)

    fittedmodel = fitbayes(model, niter=niter, warmup=warmup)

    return fittedmodel

end

"""
    gevfitbayes(data::Dict, dataid::Symbol, Covariate::Dict)

Fit a non-stationary Generalized Extreme Value (GEV) distribution under the Bayesian paradigm to the vector of data contains in the disctionary `data`under the key `dataid`.

Covariate is a dictionary containing the covariates identifyer for each parameter (μ, ϕ, ξ).

The location parameter μ is a linear function using the covariates in `data` identified by the symbols in Covariate[:μ].
The logscale parameter ϕ is a linear function using the covariates in `data` identified by the symbols in Covariate[:ϕ].
The location parameter ξ is a linear function using the covariates in `data` identified by the symbols in Covariate[:ξ].

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

# Put the data in a dictionary
data = Dict(:y => y, :x => x, :n => n)

# Put the covariate identifier in a dictionary
Covariate = Dict(:μ => [:x], :ϕ => Symbol[], :ξ => Symbol[] )

# Estimate the parameters
gevfitbayes(data, :y, Covariate=Covariate)
```

"""
function gevfitbayes(data::Dict, dataid::Symbol ; Covariate::Dict=Dict{Symbol,Vector{Symbol}}(), niter::Int=5000, warmup::Int=2000)

    # Put empty Symbol array to stationary parameters
    for k in [:μ, :ϕ, :ξ]
        if !(haskey(Covariate,k))
            Covariate[k] = Symbol[]
        end
    end

    model = BlockMaxima(data[dataid],
        locationcov = [data[s] for s in Covariate[:μ]],
        scalecov = [data[s] for s in Covariate[:ϕ]],
        shapecov = [data[s] for s in Covariate[:ξ]])

    fittedmodel = fitbayes(model, niter=niter, warmup=warmup)

    return fittedmodel

end





"""
    gpfitbayes(y::DenseArray; niter::Int=5000, warmup::Int=2000)

Fit the Generalized Pareto (GP) distribution under the Bayesian paradigm to the vector of data `y`.

A random sample from the posterior distribution is generated using the NUTS algortihm.

Only flat prior is now supported.
"""
function gpfitbayes(y::Vector{<:Real}; niter::Int=5000, warmup::Int=2000,
     threshold::Vector{<:Real}=[0], nobsperblock::Int=1)

    data = Dict(:y => y)
    dataid = :y
    Covariate = Dict(:ϕ => Symbol[], :ξ => Symbol[])
    paramindex = paramindexing(Covariate, [:ϕ, :ξ])

    model = PeaksOverThreshold(GeneralizedPareto, data, dataid, nobsperblock, Covariate, threshold, identity, identity, paramindex)

    fittedmodel = fitbayes(model, niter=niter, warmup=warmup)

    return fittedmodel

end

"""
    gpfitbayes(data::Dict, dataid::Symbol, Covariate::Dict)

Fit a non-stationary Generalized Pareto (GEV) distribution under the Bayesian paradigm to the vector of data contains in the disctionary `data`under the key `dataid`.

Covariate is a dictionary containing the covariates identifyer for each parameter (ϕ, ξ).

The logscale parameter ϕ is a linear function using the covariates in `data` identified by the symbols in Covariate[:ϕ].
The location parameter ξ is a linear function using the covariates in `data` identified by the symbols in Covariate[:ξ].

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

# Put the data in a dictionary
data = Dict(:y => y, :x => x,)

# Put the covariate identifier in a dictionary
Covariate = Dict(:ϕ => Symbol[], :ξ => Symbol[] )

# Estimate the parameters
gpfitbayes(data, :y, Covariate=Covariate)
```

"""
function gpfitbayes(data::Dict, dataid::Symbol ;
    Covariate::Dict=Dict{Symbol,Vector{Symbol}}(), niter::Int=5000,
    warmup::Int=2000, threshold::Vector{<:Real}=[0], nobsperblock::Int=1)

    # Put empty Symbol array to stationary parameters
    for k in [:ϕ, :ξ]
        if !(haskey(Covariate,k))
            Covariate[k] = Symbol[]
        end
    end

    paramindex = paramindexing(Covariate, [:ϕ, :ξ])

    logscalefun = computeparamfunction(data, Covariate[:ϕ])
    shapefun = computeparamfunction(data, Covariate[:ξ])

    model = PeaksOverThreshold(GeneralizedPareto, data, dataid, nobsperblock, Covariate, threshold, logscalefun, shapefun, paramindex)

    fittedmodel = fitbayes(model, niter=niter, warmup=warmup)

    return fittedmodel

end







"""
    fitbayes(model::EVA; niter::Int=5000, warmup::Int=2000)

Fits the extreme-value model under the Bayesian paradigm.
"""
function fitbayes(model::EVA; niter::Int=5000, warmup::Int=2000)

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

    # paramnames = String[]
    # m = length(model.paramindex[:μ])
    # append!(paramnames, ["β₁[$i]" for i=1:m])
    # m = length(model.paramindex[:ϕ])
    # append!(paramnames, ["β₂[$i]" for i=1:m])
    # m = length(model.paramindex[:ξ])
    # append!(paramnames, ["β₃[$i]" for i=1:m])

    # sim = Chains(niter, model.nparameter, start = (warmup + 1), names = paramnames)
    sim = Chains(niter, nparameter(model), start = (warmup + 1))
    θ = NUTSVariate(initialvalues, logfgrad)
    # θ = AMWGVariate(initialvalues, 1.0, logf)
    @showprogress for i in 1:niter
        sample!(θ, adapt = (i <= warmup))
        if i > warmup
            sim[i, :, 1] = θ
        end
    end

    fittedmodel = BayesianEVA(model, sim)

    return fittedmodel

end
