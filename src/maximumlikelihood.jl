"""
    gevfit(y::DenseVector)

Fit the Generalized Extreme Value (GEV) distribution by maximum likelihood to the vector of data `y.
"""
function gevfit(y::Vector{<:Real})

    data = Dict(:y => y)
    dataid = :y
    Covariate = Dict(:μ => Symbol[], :ϕ => Symbol[], :ξ => Symbol[])
    paramindex = paramindexing(Covariate, [:μ, :ϕ, :ξ])
    nparameter = 3 + getcovariatenumber(Covariate, [:μ, :ϕ, :ξ])

    model = BlockMaxima(GeneralizedExtremeValue, data, dataid, Covariate, identity, identity, identity, nparameter, paramindex)

    fittedmodel = fit(model)

    return fittedmodel

end

"""
    gevfit(data::Dict, dataid::Symbol, Covariate::Dict)

Fit a non-stationary Generalized Extreme Value (GEV) distribution by maximum likelihood to the vector of data contains in the disctionary `data`under the key `dataid`.

Covariate is a dictionary containing the covariates identifyer for each parameter (μ, ϕ, ξ).

The location parameter μ is a linear function using the covariates in `data` identified by the symbols in Covariate[:μ].
The logscale parameter ϕ is a linear function using the covariates in `data` identified by the symbols in Covariate[:ϕ].
The location parameter ξ is a linear function using the covariates in `data` identified by the symbols in Covariate[:ξ].

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

# Put the data in a dictionary
data = Dict(:y => y, :x => x, :n => n)

# Put the covariate identifier in a dictionary
Covariate = Dict(:μ => [:x], :ϕ => Symbol[], :ξ => Symbol[] )

# Estimate the parameters
gevfit(data, :y, Covariate=Covariate)
```

The covariate may be standardized to facilitate the estimation.

"""
function gevfit(data::Dict, dataid::Symbol ;
    Covariate::Dict=Dict{Symbol,Vector{Symbol}}())

    # Put empty Symbol array to stationary parameters
    for k in [:μ, :ϕ, :ξ]
        if !(haskey(Covariate,k))
            Covariate[k] = Symbol[]
        end
    end

    paramindex = paramindexing(Covariate, [:μ, :ϕ, :ξ])
    nparameter = 3 + getcovariatenumber(Covariate, [:μ, :ϕ, :ξ])

    locationfun = computeparamfunction(data, Covariate[:μ])
    logscalefun = computeparamfunction(data, Covariate[:ϕ])
    shapefun = computeparamfunction(data, Covariate[:ξ])

    model = BlockMaxima(GeneralizedExtremeValue, data, dataid, Covariate,
        locationfun, logscalefun, shapefun, nparameter, paramindex)

    fittedmodel = fit(model)

    return fittedmodel

end

"""
    gevfit(model::EVA)

Fit the non-stationary Generalized Extreme Value (GEV) distribution by maximum likelihood of the EVA model `model`.

"""
function gevfit(model::EVA)

    fit(model)

end

"""
    gpfit(y::Vector{<:Real})

Fit the Generalized Extreme Value (GEV) distribution by maximum likelihood to the vector of data `y.
"""
function gpfit(y::Vector{<:Real}; threshold::Vector{<:Real}=[0], nobsperblock::Int=1)

    data = Dict(:y => y)
    dataid = :y
    Covariate = Dict(:ϕ => Symbol[], :ξ => Symbol[])
    paramindex = paramindexing(Covariate, [:ϕ, :ξ])
    nparameter = 2 + getcovariatenumber(Covariate, [:ϕ, :ξ])

    model = PeaksOverThreshold(GeneralizedPareto, data, dataid, nobsperblock, Covariate, threshold, identity, identity, nparameter, paramindex)

    fittedmodel = fit(model)

    return fittedmodel

end


function gpfit(data::Dict, dataid::Symbol ; Covariate::Dict=Dict{Symbol,Vector{Symbol}}(), threshold::Vector{<:Real}=[0], nobsperblock::Int=1)

    # Put empty Symbol array to stationary parameters
    for k in [:ϕ, :ξ]
        if !(haskey(Covariate,k))
            Covariate[k] = Symbol[]
        end
    end

    paramindex = paramindexing(Covariate, [:ϕ, :ξ])
    nparameter = 2 + getcovariatenumber(Covariate, [:ϕ, :ξ])

    logscalefun = computeparamfunction(data, Covariate[:ϕ])
    shapefun = computeparamfunction(data, Covariate[:ξ])

    model = PeaksOverThreshold(GeneralizedPareto, data, dataid, nobsperblock, Covariate, threshold, logscalefun, shapefun, nparameter, paramindex)

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
