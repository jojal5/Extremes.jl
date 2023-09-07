struct BlockMaxima{T} <: AbstractExtremeValueModel
    data::Variable
    location::paramfun
    logscale::paramfun
    shape::paramfun
end

include(joinpath("blockmaxima", "blockmaxima{GeneralizedExtremeValue}.jl"))
include(joinpath("blockmaxima", "blockmaxima{Gumbel}.jl"))
