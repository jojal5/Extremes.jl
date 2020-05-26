module Extremes

using Distributions, DataFrames, Dates
using Optim, NLSolversBase
using SpecialFunctions, LinearAlgebra

import CSV
import Distributions.GeneralizedExtremeValue

include("functions.jl")
include("mle_functions.jl")
include("bayes_functions.jl")
include("data_functions.jl")

export getcluster, gevfit, gpdfit, gevfitbayes, gpdfitbayes, load

end # module
