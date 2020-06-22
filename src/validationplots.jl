"""
    ecdf(y::Vector{<:Real})

Compute the empirical cumulative distribution function using the Gumbel formula (Makkonen, 2006).

*Reference:*
Makkonen, L. (2006). Plotting positions in extreme value analysis. Journal of Applied Meteorology and Climatology, 45(2), 334-340.
"""
function ecdf(y::Vector{<:Real}) # TODO : Return value
    ys = sort(y)
    n = length(ys)
    p = collect(1:n)/(n+1)

    return ys, p
end

include(joinpath("validationplots", "plots.jl"))
include(joinpath("validationplots", "plots_std.jl"))
