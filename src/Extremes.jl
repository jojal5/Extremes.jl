module Extremes

using Distributions, DataFrames, Dates
using Optim, NLSolversBase, ForwardDiff
using SpecialFunctions, LinearAlgebra, Statistics
using Mamba, ProgressMeter
using Gadfly

import CSV
import Distributions.quantile
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
    fittedEVA,

    # Extreme value analysis type
    BlockMaxima,
    ThresholdExceedance,

    # Fitted extreme value analysis model
    pwmEVA,
    MaximumLikelihoodEVA,
    BayesianEVA,

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
    cint,

<<<<<<< HEAD
    # Other functions
    returnlevel, parametervar,
=======
    # Return level
    ReturnLevel,
    returnlevel,
>>>>>>> dev

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
