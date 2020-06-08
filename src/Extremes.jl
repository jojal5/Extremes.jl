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
    BlockMaxima(data::Vector{<:Real};
        locationcov::Vector{ExplanatoryVariable} = Vector{ExplanatoryVariable}(),
        scalecov::Vector{ExplanatoryVariable} = Vector{ExplanatoryVariable}(),
        shapecov::Vector{ExplanatoryVariable} = Vector{ExplanatoryVariable}())::BlockMaxima

Creates a BlockMaxima structure.

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

struct ThresholdExceedance <: EVA
    data::Vector{<:Real}
    logscale::paramfun
    shape::paramfun
end

#TODO : RETURN LEVEL
#threshold::Vector{<:Real}
#nobservation::Int
#nobsperblock::Int

"""
    ThresholdExceedance(exceedances::Vector{<:Real};
        scalecov::Vector{ExplanatoryVariable} = Vector{ExplanatoryVariable}(),
        shapecov::Vector{ExplanatoryVariable} = Vector{ExplanatoryVariable}())::ThresholdExceedance

Creates a ThresholdExceedance structure.

"""
function ThresholdExceedance(exceedances::Vector{<:Real};
    scalecov::Vector{ExplanatoryVariable} = Vector{ExplanatoryVariable}(),
    shapecov::Vector{ExplanatoryVariable} = Vector{ExplanatoryVariable}())::ThresholdExceedance

    logscalefun = computeparamfunction(scalecov)
    shapefun = computeparamfunction(shapecov)

    return ThresholdExceedance(exceedances, paramfun(scalecov, logscalefun), paramfun(shapecov, shapefun))

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
    ThresholdExceedance,

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
