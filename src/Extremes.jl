module Extremes

using Distributions, DataFrames, Dates
using Optim, NLSolversBase, ForwardDiff
using SpecialFunctions, LinearAlgebra, Statistics
using Mamba, ProgressMeter

import CSV
import Distributions.quantile
import Statistics.var


abstract type EVA end

struct ExplanatoryVariable
    name::String
    value::Vector{<:Real}
end

struct paramfun
    covariate::Vector{ExplanatoryVariable}
    fun::Function
end

struct BlockMaxima <: EVA
    data::Vector{<:Real}
    location::paramfun
    logscale::paramfun
    shape::paramfun
end

"""
Creates a BlockMaxima structure
"""
function BlockMaxima(data::Vector{<:Real};
    locationcov::Vector{ExplanatoryVariable} = Vector{ExplanatoryVariable}(),
    scalecov::Vector{ExplanatoryVariable} = Vector{ExplanatoryVariable}(),
    shapecov::Vector{ExplanatoryVariable} = Vector{ExplanatoryVariable}())::BlockMaxima

    locationfun = computeparamfunction(locationcov)
    logscalefun = computeparamfunction(scalecov)
    shapefun = computeparamfunction(shapecov)

    return BlockMaxima(data, paramfun(locationcov, locationfun), paramfun(scalecov, logscalefun), paramfun(shapecov, shapefun))
end

struct ThresholdExceedance
    data::Vector{<:Real}
    logscale::paramfun
    shape::paramfun
end

struct PeaksOverThreshold <: EVA
    mark::ThresholdExceedance
    threshold::Vector{<:Real}
    nobservation::Int
    nobsperblock::Int
end

"""
Creates a PeaksOverThreshold structure
"""
function PeaksOverThreshold(exceedances::Vector{<:Real}, nobservation::Int;
    scalecov::Vector{ExplanatoryVariable} = Vector{ExplanatoryVariable}(),
    shapecov::Vector{ExplanatoryVariable} = Vector{ExplanatoryVariable}(),
    threshold::Vector{<:Real}=[0],
    nobsperblock::Int=1)::PeaksOverThreshold

    logscalefun = computeparamfunction(scalecov)
    shapefun = computeparamfunction(shapecov)
    te = ThresholdExceedance(exceedances, paramfun(scalecov, logscalefun), paramfun(shapecov, shapefun))

    return PeaksOverThreshold(te, threshold, nobservation, nobsperblock)
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
    H::Array{Float64, 2}
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

    # Explanatory variable type
    ExplanatoryVariable,

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
