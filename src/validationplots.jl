"""
    ecdf(y::Vector{<:Real})

Compute the empirical cumulative distribution function using the Gumbel formula (Makkonen, 2006).

*Reference:*
Makkonen, L. (2006). Plotting positions in extreme value analysis. Journal of Applied Meteorology and Climatology, 45(2), 334-340.
"""
function ecdf(y::Vector{<:Real})::Tuple{Vector{<:Real}, Vector{<:Real}}
    ys = sort(y)
    n = length(ys)
    p = collect(1:n)/(n+1)

    return ys, p
end

"""
# TODO : desc
"""
function checkstationarity(m::EVA)

    if getcovariatenumber(m) > 0
        @info "The graph is optimized for stationary models and the model provided is not."
    end

end

"""
# TODO : desc
"""
function checknonstationarity(m::EVA)

    if getcovariatenumber(m) == 0
        @info "The graph is optimized for non-stationary models and the model provided is not."
    end

end

include(joinpath("validationplots", "plots.jl"))
include(joinpath("validationplots", "plots_data.jl"))
include(joinpath("validationplots", "plots_std_data.jl"))
