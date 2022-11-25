"""
    gevfitgmle()

Estimate the GEV parameters by generalized maximum likelihood estimation.

# Implementation

Estimation with the Generalized maximum likelihood, as described by [Martins and Stedinger (2000)](https://doi.org/10.1029/1999WR900330). The function uses Nelder-Mead solver implemented in the [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl)
package to find the point where the log-likelihood is maximal.

The GEV location and logscale parameters can be modeled as function of covariates as follows:
```math
μ = X₁ × β₁,
```
```math
ϕ = X₂ × β₂,
```

Non-stationarity is not supported for the shape parameter. 
The covariates are standardized before estimating the parameters to help fit the model. They are transformed back on their original scales before returning the fitted model.

See also [`gevfitgmle`](@ref) for the other methods, [`gevfit`](@ref), [`gevfitpwm`](@ref), [`gevfitbayes`](@ref) and [`BlockMaxima`](@ref).

"""
function gevfitgmle end

"""
    gevfitgmle(y,
        shapeprior = LocationScale(-.5, 1, Beta(6, 9)),
        locationcov = Vector{Variable}(),
        logscalecov = Vector{Variable}()
        )

Estimate the GEV parameters by generalized maximum likelihood estimation.

# Arguments

- `y::Vector{<:Real}`: The vector of block maxima.
- `shapeprior::Distribution = LocationScale(-.5, 1, Beta(6, 9))`: The prior distribution of the shape parameter.
- `locationcov::Vector{<:DataItem} = Vector{Variable}()`: The covariates of the location parameter.
- `logscalecov::Vector{<:DataItem} = Vector{Variable}()`: The covariates of the log-scale parameter.

"""
function gevfitgmle(y::Vector{<:Real};
    shapeprior::Distribution=LocationScale(-.5, 1, Beta(6, 9)),
    locationcov::Vector{<:DataItem} = Vector{Variable}(),
    logscalecov::Vector{<:DataItem} = Vector{Variable}())::GeneralizedMaximumLikelihoodEVA

    locationcovstd = standardize.(locationcov)
    logscalecovstd = standardize.(logscalecov)

    model = BlockMaxima(Variable("y", y), locationcov = locationcovstd, logscalecov = logscalecovstd)

    fittedmodel = fitgmle(model, shapeprior = shapeprior)

    return transform(fittedmodel)

end

"""
    gevfitgmle(y,
        initialvalues,
        shapeprior = LocationScale(-.5, 1, Beta(6, 9)),
        locationcov = Vector{Variable}(),
        logscalecov = Vector{Variable}()
        )

Estimate the GEV parameters by generalized maximum likelihood estimation.

# Arguments

- `y::Vector{<:Real}`: the vector of block maxima.
- `initialvalues::Vector{<:Real}`: Vector of parameters initial values.
- `shapeprior::Distribution = LocationScale(-.5, 1, Beta(6, 9))`: The prior distribution of the shape parameter.
- `locationcov::Vector{<:DataItem} = Vector{Variable}()`: The covariates of the location parameter.
- `logscalecov::Vector{<:DataItem} = Vector{Variable}()`: The covariates of the log-scale parameter.

"""
function gevfitgmle(y::Vector{<:Real}, initialvalues::Vector{<:Real};
    shapeprior::Distribution=LocationScale(-.5, 1, Beta(6, 9)),
    locationcov::Vector{<:DataItem} = Vector{Variable}(),
    logscalecov::Vector{<:DataItem} = Vector{Variable}())::GeneralizedMaximumLikelihoodEVA

    model = BlockMaxima(Variable("y", y), locationcov = locationcov, logscalecov = logscalecov)

    return fitgmle(model, initialvalues, shapeprior = shapeprior)

end

"""
    gevfit(df::DataFrame,
        datacol::Symbol,
        shapeprior = LocationScale(-.5, 1, Beta(6, 9)),
        locationcovid = Vector{Symbol}(),
        logscalecovid = Vector{Symbol}()
        )

Estimate the GEV parameters by generalized maximum likelihood estimation.

# Arguments

- `df::DataFrame`: The dataframe containing the data.
- `datacol::Symbol`: The symbol of the column of `df` containing the block maxima data.
- `shapeprior::Distribution = LocationScale(-.5, 1, Beta(6, 9))`: The prior distribution of the shape parameter.
- `locationcovid::Vector{Symbol} = Vector{Symbol}()`: The symbols of the columns of `df` containing the covariates of the location parameter.
- `logscalecovid::Vector{Symbol} = Vector{Symbol}()`: The symbols of the columns of `df` containing the covariates of the log-scale parameter.

"""
function gevfitgmle(df::DataFrame, datacol::Symbol;
    shapeprior::Distribution=LocationScale(-.5, 1, Beta(6, 9)),
    locationcovid::Vector{Symbol}=Symbol[],
    logscalecovid::Vector{Symbol}=Symbol[])::GeneralizedMaximumLikelihoodEVA

    locationcovstd = standardize.(buildVariables(df, locationcovid))
    logscalecovstd = standardize.(buildVariables(df, logscalecovid))

    model = BlockMaxima(Variable(string(datacol), df[:, datacol]), locationcov = locationcovstd, logscalecov = logscalecovstd)

    fittedmodel = fitgmle(model, shapeprior = shapeprior)

    return transform(fittedmodel)

end

"""
    gevfitgmle(df::DataFrame,
        datacol::Symbol,
        initialvalues::Vector{<:Real},
        shapeprior = LocationScale(-.5, 1, Beta(6, 9)),
        locationcovid = Vector{Symbol}(),
        logscalecovid = Vector{Symbol}()
        )

Estimate the GEV parameters by generalized maximum likelihood estimation.

# Arguments

- `df::DataFrame`: The dataframe containing the data.
- `datacol::Symbol`: The symbol of the column of `df` containing the block maxima data.
- `initialvalues::Vector{<:Real}`: Vector of parameters initial values.
- `shapeprior::Distribution = LocationScale(-.5, 1, Beta(6, 9))`: The prior distribution of the shape parameter.
- `locationcovid::Vector{Symbol} = Vector{Symbol}()`: The symbols of the columns of `df` containing the covariates of the location parameter.
- `logscalecovid::Vector{Symbol} = Vector{Symbol}()`: The symbols of the columns of `df` containing the covariates of the log-scale parameter.

"""
function gevfitgmle(df::DataFrame, datacol::Symbol, initialvalues::Vector{<:Real};
    shapeprior::Distribution=LocationScale(-.5, 1, Beta(6, 9)),
    locationcovid::Vector{Symbol}=Symbol[],
    logscalecovid::Vector{Symbol}=Symbol[])::GeneralizedMaximumLikelihoodEVA

    locationcov = buildVariables(df, locationcovid)
    logscalecov = buildVariables(df, logscalecovid)

    model = BlockMaxima(Variable(string(datacol), df[:, datacol]), locationcov = locationcov, logscalecov = logscalecov)

    return fitgmle(model, initialvalues, shapeprior = shapeprior)

end

"""
    gevfitgmle(model::BlockMaxima, initialvalues::Vector{<:Real})

Estimate the parameters of the `BlockMaxima` model using the given initialvalues.

"""
function gevfitgmle(model::BlockMaxima, initialvalues::Vector{<:Real}; shapeprior::Distribution=LocationScale(-.5, 1, Beta(6, 9)))::GeneralizedMaximumLikelihoodEVA

    fitgmle(model, initialvalues, shapeprior = shapeprior)

end