"""
    gevfit()

Estimate the GEV parameters by maximum likelihood.

# Implementation

The function uses Nelder-Mead solver implemented in the [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl)
package to find the point where the log-likelihood is maximal.

The GEV parameters can be modeled as function of covariates as follows:
```math
μ = X₁ × β₁,
```
```math
ϕ = X₂ × β₂,
```
```math
ξ = X₃ × β₃.
```

The covariates are standardized before estimating the parameters to help fit the
 model. They are transformed back on their original scales before returning the
 fitted model.

See also [`gevfit`](@ref) for the other methods, [`gevfitpwm`](@ref), [`gevfitbayes`](@ref) and [`BlockMaxima`](@ref).

"""
function gevfit end

"""
    gevfit(y,
        locationcov = Vector{Variable}(),
        logscalecov = Vector{Variable}(),
        shapecov = Vector{Variable}()
        )

Estimate the GEV parameters.

# Arguments

- `y::Vector{<:Real}`: The vector of block maxima.
- `locationcov::Vector{<:DataItem} = Vector{Variable}()`: The covariates of the location parameter.
- `logscalecov::Vector{<:DataItem} = Vector{Variable}()`: The covariates of the log-scale parameter.
- `shapecov::Vector{<:DataItem} = Vector{Variable}()`: The covariates of the shape parameter.
"""
function gevfit(y::Vector{<:Real};
    locationcov::Vector{<:DataItem} = Vector{Variable}(),
    logscalecov::Vector{<:DataItem} = Vector{Variable}(),
    shapecov::Vector{<:DataItem} = Vector{Variable}())::MaximumLikelihoodAbstractExtremeValueModel

    locationcovstd = standardize.(locationcov)
    logscalecovstd = standardize.(logscalecov)
    shapecovstd = standardize.(shapecov)

    model = BlockMaxima{GeneralizedExtremeValue}(Variable("y", y), locationcov = locationcovstd, logscalecov = logscalecovstd, shapecov = shapecovstd)

    fittedmodel = fit(model)

    return transform(fittedmodel)

end

"""
    gevfit(y,
        initialvalues,
        locationcov = Vector{Variable}(),
        logscalecov = Vector{Variable}(),
        shapecov = Vector{Variable}()
        )

Estimate the GEV parameters.

# Arguments

- `y::Vector{<:Real}`: the vector of block maxima.
- `initialvalues::Vector{<:Real}`: Vector of parameters initial values.
- `locationcov::Vector{<:DataItem} = Vector{Variable}()`: The covariates of the location parameter.
- `logscalecov::Vector{<:DataItem} = Vector{Variable}()`: The covariates of the log-scale parameter.
- `shapecov::Vector{<:DataItem} = Vector{Variable}()`: The covariates of the shape parameter.
"""
function gevfit(y::Vector{<:Real}, initialvalues::Vector{<:Real};
    locationcov::Vector{<:DataItem} = Vector{Variable}(),
    logscalecov::Vector{<:DataItem} = Vector{Variable}(),
    shapecov::Vector{<:DataItem} = Vector{Variable}(),)::MaximumLikelihoodAbstractExtremeValueModel

    model = BlockMaxima{GeneralizedExtremeValue}(Variable("y", y), locationcov = locationcov, logscalecov = logscalecov, shapecov = shapecov)

    return fit(model, initialvalues)

end

"""
    gevfit(df::DataFrame,
        datacol::Symbol,
        locationcovid = Vector{Symbol}(),
        logscalecovid = Vector{Symbol}(),
        shapecovid = Vector{Symbol}()
        )

Estimate the GEV parameters.

# Arguments
- `df::DataFrame`: The dataframe containing the data.
- `datacol::Symbol`: The symbol of the column of `df` containing the block maxima data.
- `locationcovid::Vector{Symbol} = Vector{Symbol}()`: The symbols of the columns of `df` containing the covariates of the location parameter.
- `logscalecovid::Vector{Symbol} = Vector{Symbol}()`: The symbols of the columns of `df` containing the covariates of the log-scale parameter.
- `shapecovid::Vector{Symbol} = Vector{Symbol}()`: The symbols of the columns of `df` containing the covariates of the shape parameter.
"""
function gevfit(df::DataFrame, datacol::Symbol;
    locationcovid::Vector{Symbol}=Symbol[],
    logscalecovid::Vector{Symbol}=Symbol[],
    shapecovid::Vector{Symbol}=Symbol[])::MaximumLikelihoodAbstractExtremeValueModel

    locationcovstd = standardize.(buildVariables(df, locationcovid))
    logscalecovstd = standardize.(buildVariables(df, logscalecovid))
    shapecovstd = standardize.(buildVariables(df, shapecovid))

    model = BlockMaxima{GeneralizedExtremeValue}(Variable(string(datacol), df[:, datacol]), locationcov = locationcovstd, logscalecov = logscalecovstd, shapecov = shapecovstd)

    fittedmodel = fit(model)

    return transform(fittedmodel)

end

"""
    gevfit(df::DataFrame,
        datacol::Symbol,
        locationcovid = Vector{Symbol}(),
        logscalecovid = Vector{Symbol}(),
        shapecovid = Vector{Symbol}()
        )

Estimate the GEV parameters.

# Arguments
- `df::DataFrame`: The dataframe containing the data.
- `datacol::Symbol`: The symbol of the column of `df` containing the block maxima data.
- `initialvalues::Vector{<:Real}`: Vector of parameters initial values.
- `locationcovid::Vector{Symbol} = Vector{Symbol}()`: The symbols of the columns of `df` containing the covariates of the location parameter.
- `logscalecovid::Vector{Symbol} = Vector{Symbol}()`: The symbols of the columns of `df` containing the covariates of the log-scale parameter.
- `shapecovid::Vector{Symbol} = Vector{Symbol}()`: The symbols of the columns of `df` containing the covariates of the shape parameter.
"""
function gevfit(df::DataFrame, datacol::Symbol, initialvalues::Vector{<:Real};
    locationcovid::Vector{Symbol}=Symbol[],
    logscalecovid::Vector{Symbol}=Symbol[],
    shapecovid::Vector{Symbol}=Symbol[])::MaximumLikelihoodAbstractExtremeValueModel

    locationcov = buildVariables(df, locationcovid)
    logscalecov = buildVariables(df, logscalecovid)
    shapecov = buildVariables(df, shapecovid)

    model = BlockMaxima{GeneralizedExtremeValue}(Variable(string(datacol), df[:, datacol]), locationcov = locationcov, logscalecov = logscalecov, shapecov = shapecov)

    return fit(model, initialvalues)

end

"""
    gevfit(model::{BlockMaxima{GeneralizedExtremeValue}}, initialvalues::Vector{<:Real})

Estimate the parameters of the `BlockMaxima` model using the given initialvalues.
"""
function gevfit(model::BlockMaxima{GeneralizedExtremeValue}, initialvalues::Vector{<:Real})::MaximumLikelihoodAbstractExtremeValueModel

    fit(model, initialvalues)

end
