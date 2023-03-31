module Extremes

using Distributions, DataFrames, Dates
using Optim
using LinearAlgebra, MambaLite, Statistics
using ProgressMeter
using Gadfly

import CSV
import ForwardDiff, SpecialFunctions

import Distributions: insupport, location, logpdf, maximum, minimum, params, pdf, quantile, scale, shape
import Statistics.var
import Base.length, Base.maximum, Base.sum

include("utils.jl")
include("structures.jl")
include("parameterestimation.jl")
include("data.jl")
include("validationplots.jl")

export

    # Variable type
    Variable, VariableStd, DataItem,

    # Cluster type
    Cluster,

    # Generic types
    EVA,
    AbstractFittedExtremeValueModel,

    # Extreme value analysis type
    BlockMaxima,
    ThresholdExceedance,

    # Fitted extreme value analysis model
    pwmEVA,
    MaximumLikelihoodEVA,
    BayesianEVA,

    # Data related functions
    getcluster,

    # Fitting functions
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
    returnlevel,


    # Diagnostic plots
    probplot_std_data,
    probplot_std,
    qqplot_std_data,
    qqplot_std,
    diagnosticplots_std,

    probplot_data,
    probplot,
    qqplot_data,
    qqplot,
    returnlevelplot_data,
    returnlevelplot,
    histplot_data,
    histplot,
    diagnosticplots,
    mrlplot,
    mrlplot_data

end # module
