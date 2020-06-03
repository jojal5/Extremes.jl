module Extremes

using Distributions, DataFrames, Dates
using Optim, NLSolversBase, ForwardDiff
using SpecialFunctions, LinearAlgebra, Statistics
using Mamba, ProgressMeter

import CSV
import Distributions.quantile
import Statistics.var


abstract type EVA end

struct paramfun
    covariate::Vector{Vector{T}} where T<:Real
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
    locationcov::Vector{Vector{T}} where T<:Real = Vector{Vector{Float64}}(),
    scalecov::Vector{Vector{T}} where T<:Real = Vector{Vector{Float64}}(),
    shapecov::Vector{Vector{T}} where T<:Real = Vector{Vector{Float64}}())

    locationfun = computeparamfunction(locationcov)
    logscalefun = computeparamfunction(scalecov)
    shapefun = computeparamfunction(shapecov)

    return BlockMaxima(data, paramfun(locationcov, locationfun), paramfun(scalecov, logscalefun), paramfun(shapecov, shapefun))
end

struct PeaksOverThreshold <: EVA
    data::Dict
    dataid::Symbol
    nobsperblock::Int
    covariate::Dict
    threshold::Vector{<:Real}
    logscalefun::Function
    shapefun::Function
end

"""
Creates a PeaksOverThreshold structure
"""
# TODO : Type for params
function PeaksOverThreshold(data::Vector{<:Real};
    scalecov = Vector{Vector{Float64}}(),
    shapecov = Vector{Vector{Float64}}(),
    threshold::Vector{<:Real}=[0],
    nobsperblock::Int=1)

    d = Dict(:data => data, :n => length(data))
    scaSym = Symbol[]
    shaSym = Symbol[]
    for i in 1:length(scalecov)
        s = Symbol(string("sc", i))
        push!(d, s => scalecov[i])
        push!(scaSym, s)
    end
    for i in 1:length(shapecov)
        s = Symbol(string("sh", i))
        push!(d, s => shapecov[i])
        push!(shaSym, s)
    end

    dataid = :data
    Covariate = Dict(:ϕ => scaSym,:ξ => shaSym)

    logscalefun = computeparamfunction(scalecov)
    shapefun = computeparamfunction(shapecov)

    return PeaksOverThreshold(d, dataid, nobsperblock, Covariate, threshold, logscalefun, shapefun)
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
