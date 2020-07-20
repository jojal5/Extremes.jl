"""
    gpfitbayes(..., niter::Int=5000, warmup::Int=2000)

Generate a sample from the GP parameters' posterior distribution.

Data provided must be exceedances above the threshold, *i.e.* the data minus
the threshold.

# Arguments
- `niter::Int = 5000`: The total number of MCMC iterations
- `warmup::Int = 2000`: The number of warmup iterations (burn-in).

# Implementation

The function uses the No-U-Turn Sampler (NUTS; [Hoffman and Gelman, 2014](http://jmlr.org/papers/v15/hoffman14a.html))
implemented in the [Mamba.jl](https://mambajl.readthedocs.io/en/latest/index.html)
package to generate a random sample from the posterior distribution.

Currently, only the improper uniform prior is implemented, *i.e.*
```math
f_{(β₂,β₃)}(β₂,β₃) ∝ 1,
```
where
```math
ϕ = X₂ × β₂,
```
```math
ξ = X₃ × β₃.
```
In the stationary case, this improper prior yields to a proper posterior if the
sample size is larger than 2 ([Northrop and Attalides, 2016](https://www.jstor.org/stable/24721296?seq=1)).

The covariates are standardized before estimating the parameters to help fit the
 model. They are transformed back on their original scales before returning the
 fitted model.

See also [`gpfitbayes`](@ref) for the other methods, [`gpfitpwm`](@ref) and [`ThresholdExceedance`](@ref).

# Reference

Hoffman M. D. and Gelman A. (2014). The No-U-Turn sampler: adaptively setting path lengths in Hamiltonian Monte Carlo. *Journal of Machine Learning Research*, 15:1593–1623.

Paul J. Northrop P. J. and Attalides N. (2016). Posterior propriety in Bayesian extreme value analyses using reference priors. *Statistica Sinica*, 26:721-743.

"""
function gpfitbayes end


"""
    gpfitbayes(y,
        logscalecov = Vector{Variable}(),
        shapecov::Vector{<:DataItem} = Vector{Variable}(),
        niter::Int=5000,
        warmup::Int=2000
        )::BayesianEVA

Generate a sample from the GP parameters' posterior distribution.

# Arguments

- `y::Vector{<:Real}`: the vector of exceedances.
- `logscalecov::Vector{<:DataItem} = Vector{Variable}()`: The covariates of the log-scale parameter.
- `shapecov::Vector{<:DataItem} = Vector{Variable}()`: The covariates of the shape parameter.

See also [`gpfitbayes`](@ref) for the other methods, [`gpfitpwm`](@ref) and [`ThresholdExceedance`](@ref).
"""
function gpfitbayes(y::Vector{<:Real};
     logscalecov::Vector{<:DataItem} = Vector{Variable}(),
     shapecov::Vector{<:DataItem} = Vector{Variable}(),
     niter::Int=5000, warmup::Int=2000)::BayesianEVA

     logscalecovstd = standardize.(logscalecov)
     shapecovstd = standardize.(shapecov)

    model = ThresholdExceedance(Variable("y", y), logscalecov = logscalecovstd, shapecov = shapecovstd)

    fittedmodel = fitbayes(model, niter=niter, warmup=warmup)

    return transform(fittedmodel)

end

"""
    gpfitbayes(df::DataFrame,
        datacol::Symbol,
        logscalecovid = Vector{Symbol}(),
        shapecovid = Vector{Symbol}(),
        niter::Int=5000,
        warmup::Int=2000)

Generate a sample from the GP parameters' posterior distribution.

# Arguments
- `df::DataFrame`: The dataframe containing the data.
- `datacol::Symbol`: The symbol of the column of `df` containing the exceedances.
- `logscalecovid::Vector{Symbol} = Vector{Symbol}()`: The symbols of the columns of `df` containing the covariates of the log-scale parameter.
- `shapecovid::Vector{Symbol} = Vector{Symbol}()`: The symbols of the columns of `df` containing the covariates of the shape parameter.

See also [`gpfitbayes`](@ref) for the other methods, [`gpfitpwm`](@ref) and [`ThresholdExceedance`](@ref).
"""
function gpfitbayes(df::DataFrame, datacol::Symbol;
    logscalecovid::Vector{Symbol}=Symbol[],
    shapecovid::Vector{Symbol}=Symbol[],
    niter::Int=5000, warmup::Int=2000)::BayesianEVA

    logscalecovstd = standardize.(buildVariables(df, logscalecovid))
    shapecovstd = standardize.(buildVariables(df, shapecovid))

    model = ThresholdExceedance(Variable(string(datacol), df[:, datacol]), logscalecov = logscalecovstd, shapecov = shapecovstd)

    fittedmodel = fitbayes(model, niter=niter, warmup=warmup)

    return transform(fittedmodel)

    return fm

end

"""
    gpfitbayes(model::ThresholdExceedance;
        niter::Int=5000,
        warmup::Int=2000)

Generate a sample from the GP parameters' posterior distribution.

# Arguments
- `model::ThresholdExceedance`: The `ThresholdExceedance` to fit.

See also [`gpfitbayes`](@ref) for the other methods, [`gpfitpwm`](@ref) and [`ThresholdExceedance`](@ref).
"""
function gpfitbayes(model::ThresholdExceedance; niter::Int=5000, warmup::Int=2000)::BayesianEVA

    return fitbayes(model, niter=niter, warmup=warmup)

end
