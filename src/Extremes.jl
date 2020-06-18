module Extremes

using Distributions, DataFrames, Dates
using Optim, NLSolversBase, ForwardDiff
using SpecialFunctions, LinearAlgebra, Statistics
using Mamba, ProgressMeter

import CSV
import Distributions.quantile
import Statistics.var

include("utils.jl")
include("structures.jl")
include("parameterestimation.jl")
include("data.jl")

export

    # Generic types
    EVA,
    fittedEVA,

    # Explanatory variable type
    Variable, VariableStd, DataItem,

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
