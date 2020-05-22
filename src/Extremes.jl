module Extremes

using Distributions, DataFrames, Dates
using Optim, NLSolversBase, ForwardDiff
using SpecialFunctions, LinearAlgebra
using Mamba, ProgressMeter

import Distributions.quantile
import Statistics.var

struct EVA
    distribution::Type
    data::Dict
    dataid::Symbol
    covariate::Dict
    nparameters::Int
    locationfun::Function
    logscalefun::Function
    shapefun::Function
    paramindex::Dict
end

struct MaximumLikelihoodEVA
    "Extreme value model definition"
    model::EVA
    "Maximum likelihood estimate"
    θ̂::Vector{Float64}
    "Hessian matrix"
    H::Array{Float64}
end

struct BayesianEVA
    "Extreme value model definition"
    model::EVA
    "MCMC outputs"
    sim::Mamba.Chains
end

Base.Broadcast.broadcastable(obj::Extremes.EVA) = Ref(obj)

include("functions.jl")
include("mle_functions.jl")
include("bayes_functions.jl")
include("utils.jl")

export getcluster, gevfit, gevfitbayes

end # module
