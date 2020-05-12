module Extremes

using Distributions, DataFrames, Dates
using Optim, NLSolversBase, ForwardDiff
using SpecialFunctions, LinearAlgebra
using Mamba, ProgressMeter

mutable struct EVA
    distribution::Type
    method::String
    data::Dict
    dataid::Symbol
    covariate::Dict
    nparameters::Int
    locationfun::Function
    logscalefun::Function
    shapefun::Function
    paramindex::Dict
    results
end

include("functions.jl")
include("mle_functions.jl")
include("bayes_functions.jl")

export getcluster, gevfit, gevfit!, gevfitbayes, gevfitbayes!, gpdfit, gpdfitbayes

end # module
