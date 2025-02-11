module Extremes

using Distributions, DataFrames, Dates, PDMats
using Optim
using LinearAlgebra, MambaLite, Statistics
using ProgressMeter
using Gadfly

import CSV
import ForwardDiff, SpecialFunctions

import Distributions: fit, fit_mle, insupport, location, logpdf, maximum, minimum, params, pdf, quantile, scale, shape
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
    qqplotci,
    returnlevelplot_data,
    returnlevelplot,
    returnlevelplotci,
    histplot_data,
    histplot,
    diagnosticplots,
    mrlplot,
    mrlplot_data

end # module
