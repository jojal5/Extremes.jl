"""
    gevfit(y::Vector{<:Real};
        locationcov::Vector{Vector{T}} where T<:Real = Vector{Vector{Float64}}(),
        logscalecov::Vector{Vector{T}} where T<:Real = Vector{Vector{Float64}}(),
        shapecov::Vector{Vector{T}} where T<:Real = Vector{Vector{Float64}}())

Fit the Generalized Extreme Value (GEV) distribution by maximum likelihood to the vector of data `y`.

The optional parameter `locationcov` is a vector containing the covariates for the parameter μ.
The optional parameter `logscalecov` is a vector containing the covariates for the parameter σ.
The optional parameter `shapecov` is a vector containing the covariates for the parameter ξ.

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

# Estimate the parameters
gevfit(y, locationcov = [Variable("x", x)])
```

The covariate may be standardized to facilitate the estimation.

"""
function gevfit(y::Vector{<:Real};
    locationcov::Vector{<:DataItem} = Vector{Variable}(),
    logscalecov::Vector{<:DataItem} = Vector{Variable}(),
    shapecov::Vector{<:DataItem} = Vector{Variable}())::MaximumLikelihoodEVA

    locationcovstd = standardize.(locationcov)
    logscalecovstd = standardize.(logscalecov)
    shapecovstd = standardize.(shapecov)

    model = BlockMaxima(Variable("y", y), locationcov = locationcovstd, logscalecov = logscalecovstd, shapecov = shapecovstd)

    fittedmodel = fit(model)

    return transform(fittedmodel)

end

"""
    gevfit(y::Vector{<:Real}, initialvalues::Vector{<:Real};
        locationcov::Vector{<:DataItem} = Vector{Variable}(),
        logscalecov::Vector{<:DataItem} = Vector{Variable}(),
        shapecov::Vector{<:DataItem} = Vector{Variable}(),)::MaximumLikelihoodEVA

Fit the Generalized Extreme Value (GEV) distribution by maximum likelihood to the vector of data `y` using the intial values `initialvalues`.

The covariate may be standardized to facilitate the estimation.

"""
function gevfit(y::Vector{<:Real}, initialvalues::Vector{<:Real};
    locationcov::Vector{<:DataItem} = Vector{Variable}(),
    logscalecov::Vector{<:DataItem} = Vector{Variable}(),
    shapecov::Vector{<:DataItem} = Vector{Variable}(),)::MaximumLikelihoodEVA

    model = BlockMaxima(Variable("y", y), locationcov = locationcov, logscalecov = logscalecov, shapecov = shapecov)

    return fit(model, initialvalues = initialvalues)

end

"""
    gevfit(df::DataFrame, datacol::Symbol;
        locationcovid::Vector{Symbol}=Symbol[],
        logscalecovid::Vector{Symbol}=Symbol[],
        shapecovid::Vector{Symbol}=Symbol[])::MaximumLikelihoodEVA

Fit the Generalized Extreme Value (GEV) distribution by maximum likelihood to the vector of data contained in the dataframe `df` at the column `datacol`.

"""
function gevfit(df::DataFrame, datacol::Symbol;
    locationcovid::Vector{Symbol}=Symbol[],
    logscalecovid::Vector{Symbol}=Symbol[],
    shapecovid::Vector{Symbol}=Symbol[])::MaximumLikelihoodEVA

    locationcovstd = standardize.(buildVariables(df, locationcovid))
    logscalecovstd = standardize.(buildVariables(df, logscalecovid))
    shapecovstd = standardize.(buildVariables(df, shapecovid))

    model = BlockMaxima(Variable(string(datacol), df[:, datacol]), locationcov = locationcovstd, logscalecov = logscalecovstd, shapecov = shapecovstd)

    fittedmodel = fit(model)

    return transform(fittedmodel)

end

"""
    gevfit(df::DataFrame, datacol::Symbol, initialvalues::Vector{<:Real};
        locationcovid::Vector{Symbol}=Symbol[],
        logscalecovid::Vector{Symbol}=Symbol[],
        shapecovid::Vector{Symbol}=Symbol[])::MaximumLikelihoodEVA

Fit the Generalized Extreme Value (GEV) distribution by maximum likelihood to the vector of data contained in the dataframe `df` at the column `datacol` using the initial values `ìnitialvalues`.

"""
function gevfit(df::DataFrame, datacol::Symbol, initialvalues::Vector{<:Real};
    locationcovid::Vector{Symbol}=Symbol[],
    logscalecovid::Vector{Symbol}=Symbol[],
    shapecovid::Vector{Symbol}=Symbol[])::MaximumLikelihoodEVA

    locationcov = buildVariables(df, locationcovid)
    logscalecov = buildVariables(df, logscalecovid)
    shapecov = buildVariables(df, shapecovid)

    model = BlockMaxima(Variable(string(datacol), df[:, datacol]), locationcov = locationcov, logscalecov = logscalecov, shapecov = shapecov)

    return fit(model, initialvalues = initialvalues)

end


"""
    gevfit(model::BlockMaxima)

Fit the Generalized Extreme Value (GEV) distribution by maximum likelihood of the BlockMaxima model `model`.

"""
function gevfit(model::BlockMaxima, initialvalues::Vector{<:Real})::MaximumLikelihoodEVA

    fit(model, initialvalues = initialvalues)

end
