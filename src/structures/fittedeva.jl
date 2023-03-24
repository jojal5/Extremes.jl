"""
    fittedEVA{T<:EVA}

Abstract type containing the fitted extreme value model types.

- BayesianEVA
- MaximumLikelihoodEVA
- pwmEVA

"""
abstract type fittedEVA{T<:EVA} end

Base.Broadcast.broadcastable(obj::fittedEVA) = Ref(obj)

"""
    location(fm::fittedEVA)

Return the location parameters of the fitted model.     
"""
function location(fm::fittedEVA)
   
    fd = Extremes.getdistribution(fm)
    
    return location.(fd)
    
end

"""
    params(fm::fittedEVA)

Return the parameters of the fitted model.
"""
function params(fm::fittedEVA)
    fd = Extremes.getdistribution(fm)

    return params.(fd)

 end

 """
    scale(fm::fittedEVA)

Return the scale parameters of the fitted model.
 """
 function scale(fm::fittedEVA)
   
    fd = Extremes.getdistribution(fm)
    
    return scale.(fd)
    
end

"""
    shape(fm::fittedEVA)

Return the shape parameters of the fitted model.
"""
function shape(fm::fittedEVA)
   
    fd = Extremes.getdistribution(fm)
    
    return shape.(fd)
    
end

"""
    Base.show(io::IO, obj::fittedEVA)

Override of the show function for the objects of type fittedEVA.

"""
function Base.show(io::IO, obj::fittedEVA)

    showfittedEVA(io, obj)

end

"""
    cint(..., confidencelevel::Real=.95)

Compute confidence interval or credible interval in the case of Bayesian estimation.

The function can be applied on any [`fittedEVA`](@ref) subtype to obtain a
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
include(joinpath("fittedeva", "bayesianeva.jl"))
include(joinpath("fittedeva", "maximumlikelihoodeva.jl"))
include(joinpath("fittedeva", "pwmeva.jl"))
