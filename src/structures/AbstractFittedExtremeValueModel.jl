"""
    AbstractFittedExtremeValueModel{T<:AbstractExtremeValueModel}

Abstract type containing the fitted extreme value model types.

- BayesianAbstractExtremeValueModel
- MaximumLikelihoodAbstractExtremeValueModel
- pwmAbstractExtremeValueModel

"""
abstract type AbstractFittedExtremeValueModel{T<:AbstractExtremeValueModel} end

Base.Broadcast.broadcastable(obj::AbstractFittedExtremeValueModel) = Ref(obj)

"""
    location(fm::AbstractFittedExtremeValueModel)

Return the location parameters of the fitted model.     
"""
function location(fm::AbstractFittedExtremeValueModel)
   
    fd = Extremes.getdistribution(fm)
    
    return location.(fd)
    
end

"""
    params(fm::AbstractFittedExtremeValueModel)

Return the parameters of the fitted model.
"""
function params(fm::AbstractFittedExtremeValueModel)
    fd = Extremes.getdistribution(fm)

    return params.(fd)

 end

 """
    scale(fm::AbstractFittedExtremeValueModel)

Return the scale parameters of the fitted model.
 """
 function scale(fm::AbstractFittedExtremeValueModel)
   
    fd = Extremes.getdistribution(fm)
    
    return scale.(fd)
    
end

"""
    shape(fm::AbstractFittedExtremeValueModel)

Return the shape parameters of the fitted model.
"""
function shape(fm::AbstractFittedExtremeValueModel)
   
    fd = Extremes.getdistribution(fm)
    
    return shape.(fd)
    
end

"""
    Base.show(io::IO, obj::AbstractFittedExtremeValueModel)

Override of the show function for the objects of type AbstractFittedExtremeValueModel.

"""
function Base.show(io::IO, obj::AbstractFittedExtremeValueModel)

    showAbstractFittedExtremeValueModel(io, obj)

end

"""
    cint(..., confidencelevel::Real=.95)

Compute confidence interval or credible interval in the case of Bayesian estimation.

The function can be applied on any [`AbstractFittedExtremeValueModel`](@ref) subtype to obtain a
confidence interval on the model parameters. It can also be applied on
[`ReturnLevel`](@ref) type to obtain a confidence interval on the return level.

# Implementation

The method used for computing the interval depends on the estimation method. In
the case of maximum likelihood estimation, the confidence intervals are computed
using the Wald approximation based on the approximate parameter estimates
covariance matrix. In the case of Bayesian estimation, the return interval is
the highest posterior density estimate based on the MCMC sample. In the case of
probability weighted moment estimation, the intervals are computed using a
boostrap procedure.

"""
function cint end

include("returnlevel.jl")
include(joinpath("AbstractFittedExtremeValueModel", "bayesianAbstractExtremeValueModel.jl"))
include(joinpath("AbstractFittedExtremeValueModel", "maximumlikelihoodAbstractExtremeValueModel.jl"))
include(joinpath("AbstractFittedExtremeValueModel", "pwmAbstractExtremeValueModel.jl"))
