abstract type fittedEVA end

struct ReturnLevel
      fittedmodel::fittedEVA
      returnperiod::Real
      value::Vector{<:Real}
      cint::Vector{Vector{T}} where T<:Real
end

include(joinpath("fittedeva", "bayesianeva.jl"))
include(joinpath("fittedeva", "maximumlikelihoodeva.jl"))
include(joinpath("fittedeva", "pwmeva.jl"))
