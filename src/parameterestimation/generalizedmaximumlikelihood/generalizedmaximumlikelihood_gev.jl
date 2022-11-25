"""
    gevfitgmle()
Estimate the GEV parameters by Generalized maximum likelihood.
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
function gevfitgmle end

"""
    gmlefit(y,
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
function gevfitgmle(y::Vector{<:Real};
    shapeprior::Distribution=LocationScale(-.5, 1, Beta(6, 9)),
    locationcov::Vector{<:DataItem} = Vector{Variable}(),
    logscalecov::Vector{<:DataItem} = Vector{Variable}(),
    shapecov::Vector{<:DataItem} = Vector{Variable}())::GeneralizedMaximumLikelihoodEVA

    locationcovstd = standardize.(locationcov)
    logscalecovstd = standardize.(logscalecov)
    shapecovstd = standardize.(shapecov)

    model = BlockMaxima(Variable("y", y), locationcov = locationcovstd, logscalecov = logscalecovstd, shapecov = shapecovstd)

    fittedmodel = fitgmle(model, shapeprior = shapeprior)

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
function gevfitgmle(y::Vector{<:Real}, initialvalues::Vector{<:Real};
    shapeprior::Distribution=LocationScale(-.5, 1, Beta(6, 9)),
    locationcov::Vector{<:DataItem} = Vector{Variable}(),
    logscalecov::Vector{<:DataItem} = Vector{Variable}(),
    shapecov::Vector{<:DataItem} = Vector{Variable}(),)::GeneralizedMaximumLikelihoodEVA

    model = BlockMaxima(Variable("y", y), locationcov = locationcov, logscalecov = logscalecov, shapecov = shapecov)

    return fitgmle(model, initialvalues, shapeprior = shapeprior)

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
function gevfitgmle(df::DataFrame, datacol::Symbol;
    shapeprior::Distribution=LocationScale(-.5, 1, Beta(6, 9)),
    locationcovid::Vector{Symbol}=Symbol[],
    logscalecovid::Vector{Symbol}=Symbol[],
    shapecovid::Vector{Symbol}=Symbol[])::GeneralizedMaximumLikelihoodEVA

    locationcovstd = standardize.(buildVariables(df, locationcovid))
    logscalecovstd = standardize.(buildVariables(df, logscalecovid))
    shapecovstd = standardize.(buildVariables(df, shapecovid))

    model = BlockMaxima(Variable(string(datacol), df[:, datacol]), locationcov = locationcovstd, logscalecov = logscalecovstd, shapecov = shapecovstd)

    fittedmodel = fitgmle(model, shapeprior = shapeprior)

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
function gevfitgmle(df::DataFrame, datacol::Symbol, initialvalues::Vector{<:Real};
    shapeprior::Distribution=LocationScale(-.5, 1, Beta(6, 9)),
    locationcovid::Vector{Symbol}=Symbol[],
    logscalecovid::Vector{Symbol}=Symbol[],
    shapecovid::Vector{Symbol}=Symbol[])::GeneralizedMaximumLikelihoodEVA

    locationcov = buildVariables(df, locationcovid)
    logscalecov = buildVariables(df, logscalecovid)
    shapecov = buildVariables(df, shapecovid)

    model = BlockMaxima(Variable(string(datacol), df[:, datacol]), locationcov = locationcov, logscalecov = logscalecov, shapecov = shapecov)

    return fitgmle(model, initialvalues, shapeprior = shapeprior)

end

"""
    gevfit(model::BlockMaxima, initialvalues::Vector{<:Real})
Estimate the parameters of the `BlockMaxima` model using the given initialvalues.
"""
function gevfitgmle(model::BlockMaxima, initialvalues::Vector{<:Real}; shapeprior::Distribution=LocationScale(-.5, 1, Beta(6, 9)))::GeneralizedMaximumLikelihoodEVA

    fitgmle(model, initialvalues, shapeprior = shapeprior)

end