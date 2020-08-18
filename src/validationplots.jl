"""
    ecdf(y::Vector{<:Real})::Tuple{Vector{<:Real}, Vector{<:Real}}

Compute the empirical cumulative distribution function using the Gumbel formula.

The empirical quantiles are computed using the Gumbel plotting positions as
as recommended by [Makkonen (2006)](https://journals.ametsoc.org/jamc/article/45/2/334/12668/Plotting-Positions-in-Extreme-Value-Analysis).

# Example
```julia-repl
julia> (x, F̂) = Extremes.ecdf(y)
```

# Reference
Makkonen, L. (2006). Plotting positions in extreme value analysis. Journal of
Applied Meteorology and Climatology, 45(2), 334-340.
"""
function ecdf(y::Vector{<:Real})::Tuple{Vector{<:Real}, Vector{<:Real}}
    ys = sort(y)
    n = length(ys)
    p = collect(1:n)/(n+1)

    return ys, p
end

"""
    checkstationarity(model::EVA)

Check if the extreme value model `model` is stationary.
"""
function checkstationarity(m::EVA)

    if getcovariatenumber(m) > 0
        @info "The graph is optimized for stationary models and the model provided is not."
    end

end

"""
    checknonstationarity(model::EVA)

Check if the extreme value model `model` is nonstationary.
"""
function checknonstationarity(m::EVA)

    if getcovariatenumber(m) == 0
        @info "The graph is optimized for non-stationary models and the model provided is not."
    end

end

include(joinpath("validationplots", "plots.jl"))
include(joinpath("validationplots", "plots_data.jl"))
include(joinpath("validationplots", "plots_std_data.jl"))
