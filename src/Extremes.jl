module Extremes

using Distributions, DataFrames, Dates
using Optim, NLSolversBase, ForwardDiff
using SpecialFunctions, LinearAlgebra, Statistics
using Mamba, ProgressMeter

import CSV
import Distributions.quantile
import Statistics.var


abstract type EVA end

struct BlockMaxima <: EVA
    distribution::Type
    data::Dict
    dataid::Symbol
    covariate::Dict
    locationfun::Function
    logscalefun::Function
    shapefun::Function
    nparameter::Int
    paramindex::Dict
end

struct PeaksOverThreshold <: EVA
    distribution::Type
    data::Dict
    dataid::Symbol
    nobsperblock::Int
    covariate::Dict
    threshold::Vector{<:Real}
    logscalefun::Function
    shapefun::Function
    nparameter::Int
    paramindex::Dict
end

abstract type fittedEVA end

struct pwmEVA <: fittedEVA
    "Extreme value model definition"
    model::EVA
    "Maximum likelihood estimate"
    θ̂::Vector{Float64}
end

struct MaximumLikelihoodEVA <: fittedEVA
    "Extreme value model definition"
    model::EVA
    "Maximum likelihood estimate"
    θ̂::Vector{Float64}
    "Hessian matrix"
    H::Array{Float64}
end

struct BayesianEVA <: fittedEVA
    "Extreme value model definition"
    model::EVA
    "MCMC outputs"
    sim::Mamba.Chains
end

struct ReturnLevel
      fittedmodel::fittedEVA
      returnperiod::Real
      value::Vector{<:Real}
      cint::Vector{Vector{T}} where T<:Real
end

Base.Broadcast.broadcastable(obj::Extremes.EVA) = Ref(obj)

include("bayesian.jl")
include("data.jl")
include("functions.jl")
include("maximumlikelihood.jl")
include("probabilityweightedmoment.jl")
include("utils.jl")



export

    # Generic types
    EVA,
    fittedEVA,

    # Extreme value analysis type
    BlockMaxima,
    PeaksOverThreshold,

    # Fitted extreme value analysis model
    pwmEVA,
    MaximumLikelihoodEVA,
    BayesianEVA,

    # Other types
    ReturnLevel,

    # Data related functions
    load,
    getcluster,

    # Fitting functions
    gevfit,
    gevfitbayes,
    gevfitpwm,
    gpfit,
    gpfitbayes,
    gpfitpwm,

    # Other functions
    returnlevel

end # module
