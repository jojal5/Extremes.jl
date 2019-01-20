module Extremes

using Distributions
using JuMP, Ipopt
using SpecialFunctions, LinearAlgebra

include("functions.jl")

export gevfit, gevloglike

end # module
