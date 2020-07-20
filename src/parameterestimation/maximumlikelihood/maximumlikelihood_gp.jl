"""
    gpfit(...)

Estimate the GP parameters by maximum likelihood.

Data provided must be the exceedances above the threshold, *i.e.* the data above the threshold minus
the threshold.

# Implementation

The function uses Nelder-Mead solver implemented in the [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl)
package to find the point where the log-likelihood is maximal.

The GP parameters can be modeled as function of covariates as follows:
```math
ϕ = X₂ × β₂,
```
```math
ξ = X₃ × β₃.
```

The covariates are standardized before estimating the parameters to help fit the
 model. They are transformed back on their original scales before returning the
 fitted model.

See also [`gpfit`](@ref) for the other methods, [`gpfitpwm`](@ref), [`gpfitbayes`](@ref) and [`ThresholdExceedance`](@ref).

"""
function gpfit end

"""
    gpfit(y,
        logscalecov = Vector{Variable}(),
        shapecov = Vector{Variable}()
        )

Estimate the GP parameters

# Arguments

- `y::Vector{<:Real}`: The vector of exceedances.
- `logscalecov::Vector{<:DataItem} = Vector{Variable}()`: The covariates of the log-scale parameter.
- `shapecov::Vector{<:DataItem} = Vector{Variable}()`: The covariates of the shape parameter.
"""
function gpfit(y::Vector{<:Real};
    logscalecov::Vector{<:DataItem} = Vector{Variable}(),
    shapecov::Vector{<:DataItem} = Vector{Variable}())::MaximumLikelihoodEVA

    logscalecovstd = standardize.(logscalecov)
    shapecovstd = standardize.(shapecov)

    model = ThresholdExceedance(Variable("y", y), logscalecov = logscalecovstd, shapecov = shapecovstd)

    fittedmodel = fit(model)

    return transform(fittedmodel)

end

"""
    gpfit(y,
        initialvalues;
        logscalecov = Vector{Variable}(),
        shapecov = Vector{Variable}()
        )

Estimate the GP parameters

# Arguments

- `y::Vector{<:Real}`: The vector of exceedances.
- `initialvalues::Vector{<:Real}`: The vector of parameters initial values.
- `logscalecov::Vector{<:DataItem} = Vector{Variable}()`: The covariates of the log-scale parameter.
- `shapecov::Vector{<:DataItem} = Vector{Variable}()`: The covariates of the shape parameter.
"""
function gpfit(y::Vector{<:Real}, initialvalues::Vector{<:Real};
    logscalecov::Vector{<:DataItem} = Vector{Variable}(),
    shapecov::Vector{<:DataItem} = Vector{Variable}())::MaximumLikelihoodEVA

    model = ThresholdExceedance(Variable("y", y), logscalecov = logscalecov, shapecov = shapecov)

    return fit(model, initialvalues)

end

"""
    gpfit(df::DataFrame,
        datacol::Symbol,
        logscalecovid = Vector{Symbol}(),
        shapecovid = Vector{Symbol}()
        )

Estimate the GP parameters

# Arguments
- `df::DataFrame`: The dataframe containing the data.
- `datacol::Symbol`: The symbol of the column of `df` containing the exceedances.
- `logscalecovid::Vector{Symbol} = Vector{Symbol}()`: The symbols of the columns of `df` containing the covariates of the log-scale parameter.
- `shapecovid::Vector{Symbol} = Vector{Symbol}()`: The symbols of the columns of `df` containing the covariates of the shape parameter.
"""
function gpfit(df::DataFrame, datacol::Symbol;
    logscalecovid::Vector{Symbol}=Symbol[],
    shapecovid::Vector{Symbol}=Symbol[])::MaximumLikelihoodEVA

    logscalecovstd = standardize.(buildVariables(df, logscalecovid))
    shapecovstd = standardize.(buildVariables(df, shapecovid))

    model = ThresholdExceedance(Variable(string(datacol), df[:, datacol]), logscalecov = logscalecovstd, shapecov = shapecovstd)

    fittedmodel = fit(model)

    return transform(fittedmodel)

end

"""
    gpfit(df::DataFrame,
        datacol::Symbol,
        logscalecovid = Vector{Symbol}(),
        shapecovid = Vector{Symbol}()
        )

Estimate the GP parameters

# Arguments
- `df::DataFrame`: The dataframe containing the data.
- `datacol::Symbol`: The symbol of the column of `df` containing the exceedances.
- `initialvalues::Vector{<:Real}`: Vector of parameters initial values.
- `logscalecovid::Vector{Symbol} = Vector{Symbol}()`: The symbols of the columns of `df` containing the covariates of the log-scale parameter.
- `shapecovid::Vector{Symbol} = Vector{Symbol}()`: The symbols of the columns of `df` containing the covariates of the shape parameter.
"""
function gpfit(df::DataFrame, datacol::Symbol, initialvalues::Vector{<:Real};
    logscalecovid::Vector{Symbol}=Symbol[],
    shapecovid::Vector{Symbol}=Symbol[])::MaximumLikelihoodEVA

    logscalecov = buildVariables(df, logscalecovid)
    shapecov = buildVariables(df, shapecovid)

    model = ThresholdExceedance(Variable(string(datacol), df[:, datacol]), logscalecov = logscalecov, shapecov = shapecov)

    return fit(model, initialvalues)

end

"""
    gpfit(model::ThresholdExceedance, initialvalues::Vector{<:Real})

Estimate the parameters of the `ThresholdExceedance` model using the given initialvalues.
"""
function gpfit(model::ThresholdExceedance, initialvalues::Vector{<:Real})::MaximumLikelihoodEVA

    return fit(model, initialvalues)

end
