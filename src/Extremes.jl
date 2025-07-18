module Extremes

using Distributions, DataFrames, Dates, PDMats
using Optim
using LinearAlgebra, MambaLite, Statistics
using ProgressMeter

import CSV
import ForwardDiff, SpecialFunctions

import Distributions: fit, fit_mle, insupport, location, logpdf, maximum, minimum, params, pdf, quantile, scale, shape
import Statistics.var
import Base.length, Base.maximum, Base.sum

include("utils.jl")
include("structures.jl")
include("parameterestimation.jl")
include("data.jl")

export

    # Variable type
    Variable, VariableStd, DataItem,

    # Cluster type
    Cluster,

    # Generic types
    AbstractExtremeValueModel,
    AbstractFittedExtremeValueModel,

    # Extreme value analysis type
    BlockMaxima,
    ThresholdExceedance,

    # Fitted extreme value analysis model
    pwmAbstractExtremeValueModel,
    MaximumLikelihoodAbstractExtremeValueModel,
    BayesianAbstractExtremeValueModel,

    # Data related functions
    getcluster,

    # Fitting functions
    fit,
    gevfit,
    gevfitbayes,
    gevfitpwm,
    gpfit,
    gpfitbayes,
    gpfitpwm,
    gumbelfit,
    gumbelfitbayes,
    gumbelfitpwm,
    cint,


    # Other functions
    
    aic,
    bic,
    location,
    parametervar,
    params,
    scale,
    shape,
    Flat,

    # Return level
    ReturnLevel,
    returnlevel

end # module
