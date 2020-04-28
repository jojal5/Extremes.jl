module Extremes

using Distributions, DataFrames, Dates
using Optim, NLSolversBase
using SpecialFunctions, LinearAlgebra

import Distributions.GeneralizedExtremeValue

include("functions.jl")
include("mle_functions.jl")
include("bayes_functions.jl")

export getcluster, gevfit, gpdfit, gevfitbayes, gpdfitbayes

end # module
