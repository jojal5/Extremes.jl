struct BlockMaxima{T} <: AbstractExtremeValueModel
    data::Variable
    location::paramfun
    logscale::paramfun
    shape::paramfun
end

"""
    standarddist(::BlockMaxima{T})::Distribution where T

Return the standard distribution after standardization, which is Gumbel(0,1) for the GEV.
"""
function standarddist(::BlockMaxima{T})::Distribution where T
    return Gumbel()
end

include(joinpath("blockmaxima", "blockmaxima{GeneralizedExtremeValue}.jl"))
include(joinpath("blockmaxima", "blockmaxima{Gumbel}.jl"))
