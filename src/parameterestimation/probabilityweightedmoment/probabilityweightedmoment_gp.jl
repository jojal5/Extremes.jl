
"""
    gpfitpwm(...)

Estimate the GP parameters with the probability weighted moments.

# Implementation

Estimation with the probability weighted moments, as described by [Hosking and Wallis (1987)](https://www.jstor.org/stable/1269343?seq=1),
is only possible in the stationary case.

See also [`gpfitpwm`](@ref) for the other methods, [`gpfit`](@ref), [`gpfitbayes`](@ref) and [`ThresholdExceedance`](@ref).

# Reference

Hosking, J. R. M. and Wallis, J. R. (1987). Parameter and quantile estimation for the Generalized Pareto distribution,
    *Technometrics*, 29:339-349.
"""
function gpfitpwm end


"""
    gpfitpwm(y::Vector{<:Real})

Estimate the GP parameters with the probability weighted moments.

"""
function gpfitpwm(y::Vector{<:Real})::pwmAbstractExtremeValueModel

    model = ThresholdExceedance(Variable("y", y))

    fittedmodel = gpfitpwm(model)

    return fittedmodel

end

"""
    gpfitpwm(df::DataFrame, datacol::Symbol)

Estimate the GP parameters with the probability weighted moments.

Block maxima data are in the column `datacol` of the dataframe `df`.

"""
function gpfitpwm(df::DataFrame, datacol::Symbol)::pwmAbstractExtremeValueModel

    model = ThresholdExceedance(Variable(string(datacol), df[:, datacol]))

    fittedmodel = gpfitpwm(model)

    return fittedmodel

end

"""
    gpfitpwm(model::ThresholdExceedance)

Estimate the GP parameters with the probability weighted moments.

"""
function gpfitpwm(model::ThresholdExceedance)::pwmAbstractExtremeValueModel

    model = validatestationarity(model)

    y = model.data.value

    a₀ = pwm(y,1,0,0)
    a₁ = pwm(y,1,0,1)

    σ̂ = 2*a₀*a₁/(a₀ - 2*a₁)

    k = a₀ / (a₀ - 2*a₁) - 2

    ξ̂ = - k
    ϕ̂ = log(σ̂)

    θ̂ = [ϕ̂; ξ̂]

    fm = pwmAbstractExtremeValueModel{ThresholdExceedance}(model, θ̂)

    return fm

end
