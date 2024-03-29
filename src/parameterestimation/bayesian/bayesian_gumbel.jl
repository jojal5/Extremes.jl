"""
    gumbelfitbayes(..., niter::Int=5000, warmup::Int=2000)

Generate a sample from the Gumbel parameters' posterior distribution.

# Arguments
- `niter::Int = 5000`: The total number of MCMC iterations.
- `warmup::Int = 2000`: The number of warmup iterations (burn-in).

# Implementation

The function uses the No-U-Turn Sampler (NUTS; [Hoffman and Gelman, 2014](http://jmlr.org/papers/v15/hoffman14a.html))
implemented in the [Mamba.jl](https://mambajl.readthedocs.io/en/latest/index.html)
package to generate a random sample from the posterior distribution.

Currently, only the improper uniform prior is implemented, *i.e.*
```math
f_{(β₁,β₂,β₃)}(β₁,β₂,β₃) ∝ 1,
```
where
```math
μ = X₁ × β₁,
```
```math
ϕ = X₂ × β₂,
```

In the stationary case, this improper prior yields to a proper posterior if the
sample size is larger than 3 ([Northrop and Attalides, 2016](https://www.jstor.org/stable/24721296?seq=1)).

The covariates are standardized before estimating the parameters to help fit the
 model. They are transformed back on their original scales before returning the
 fitted model.

See also [`gevfitbayes`](@ref) for the other methods and [`BlockMaxima`](@ref).

# References

Hoffman M. D. and Gelman A. (2014). The No-U-Turn sampler: adaptively setting path lengths in Hamiltonian Monte Carlo. *Journal of Machine Learning Research*, 15:1593–1623.

Paul J. Northrop P. J. and Attalides N. (2016). Posterior propriety in Bayesian extreme value analyses using reference priors. *Statistica Sinica*, 26:721-743.

"""
function gumbelfitbayes end

"""
    gumbelfitbayes(y,
        locationcov = Vector{Variable}(),
        logscalecov = Vector{Variable}(),
        niter::Int=5000,
        warmup::Int=2000
        )

Generate a sample from the Gumbel parameters' posterior distribution.

# Arguments

- `y::Vector{<:Real}`: The vector of block maxima.
- `locationcov::Vector{<:DataItem} = Vector{Variable}()`: The covariates of the location parameter.
- `logscalecov::Vector{<:DataItem} = Vector{Variable}()`: The covariates of the log-scale parameter.

"""
function gumbelfitbayes(y::Vector{<:Real};
    locationcov::Vector{<:DataItem} = Vector{Variable}(),
    logscalecov::Vector{<:DataItem} = Vector{Variable}(),
    niter::Int=5000, warmup::Int=2000)::BayesianAbstractExtremeValueModel

    locationcovstd = standardize.(locationcov)
    logscalecovstd = standardize.(logscalecov)

    model = BlockMaxima{Gumbel}(Variable("y", y), locationcov = locationcovstd, logscalecov = logscalecovstd)

    fittedmodel = fitbayes(model, niter=niter, warmup=warmup)

    return transform(fittedmodel)

end

"""
    gumbelfitbayes(df::DataFrame,
        datacol::Symbol,
        locationcovid = Vector{Symbol}(),
        logscalecovid = Vector{Symbol}(),
        niter::Int=5000,
        warmup::Int=2000)

Generate a sample from the Gumbel parameters' posterior distribution.

# Arguments
- `df::DataFrame`: The dataframe containing the data.
- `datacol::Symbol`: The symbol of the column of `df` containing the block maxima data.
- `locationcovid::Vector{Symbol} = Vector{Symbol}()`: The symbols of the columns of `df` containing the covariates of the location parameter.
- `logscalecovid::Vector{Symbol} = Vector{Symbol}()`: The symbols of the columns of `df` containing the covariates of the log-scale parameter.
"""
function gumbelfitbayes(df::DataFrame, datacol::Symbol;
    locationcovid::Vector{Symbol}=Symbol[],
    logscalecovid::Vector{Symbol}=Symbol[],
    niter::Int=5000, warmup::Int=2000)::BayesianAbstractExtremeValueModel

    locationcovstd = standardize.(buildVariables(df, locationcovid))
    logscalecovstd = standardize.(buildVariables(df, logscalecovid))

    model = BlockMaxima{Gumbel}(Variable(string(datacol), df[:, datacol]), locationcov = locationcovstd, logscalecov = logscalecovstd)

    fittedmodel = fitbayes(model, niter=niter, warmup=warmup)

    return transform(fittedmodel)

end

"""
    gumbelfitbayes(model::BlockMaxima{Gumbel};
        niter::Int=5000,
        warmup::Int=2000)

Generate a sample from the `BlockMaxima` model parameters' posterior distribution.
"""
function gumbelfitbayes(model::BlockMaxima{Gumbel}; niter::Int=5000, warmup::Int=2000)::BayesianAbstractExtremeValueModel

    return fitbayes(model, niter=niter, warmup=warmup)

end
