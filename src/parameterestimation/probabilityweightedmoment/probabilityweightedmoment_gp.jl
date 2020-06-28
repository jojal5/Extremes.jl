"""
    gpfitpwm(y::Vector{<:Real})::pwmEVA

Estimate the Generalized Pareto distribution parameters with the probability weighted moments as described
in Hosking & Wallis (1987).

*Reference:*
Hosking, J. R. M. and Wallis, J. R. (1987). Parameter and Quantile Estimation for the Generalized Pareto Distribution,
    Technometrics, 29(3), 339-349.

"""
function gpfitpwm(y::Vector{<:Real})::pwmEVA

    model = ThresholdExceedance(Variable("y", y))

    fittedmodel = gpfitpwm(model)

    return fittedmodel

end

"""
    gpfitpwm(df::DataFrame, datacol::Symbol)::pwmEVA

Estimate the Generalized Pareto distribution parameters with the probability weighted moments as described
in Hosking & Wallis (1987).

*Reference:*
Hosking, J. R. M. and Wallis, J. R. (1987). Parameter and Quantile Estimation for the Generalized Pareto Distribution,
    Technometrics, 29(3), 339-349.

"""
function gpfitpwm(df::DataFrame, datacol::Symbol)::pwmEVA

    model = ThresholdExceedance(Variable(string(datacol), df[:, datacol]))

    fittedmodel = gpfitpwm(model)

    return fittedmodel

end

"""
    gpfitpwm(model::ThresholdExceedance)::pwmEVA

Estimate the Generalized Pareto distribution parameters with the probability weighted moments as described
in Hosking & Wallis (1987).

*Reference:*
Hosking, J. R. M. and Wallis, J. R. (1987). Parameter and Quantile Estimation for the Generalized Pareto Distribution,
    Technometrics, 29(3), 339-349.

"""
function gpfitpwm(model::ThresholdExceedance)::pwmEVA

    model = validatestationarity(model)

    y = model.data.value

    a₀ = pwm(y,1,0,0)
    a₁ = pwm(y,1,0,1)

    σ̂ = 2*a₀*a₁/(a₀ - 2*a₁)

    k = a₀ / (a₀ - 2*a₁) - 2

    ξ̂ = - k
    ϕ̂ = log(σ̂)

    θ̂ = [ϕ̂; ξ̂]

    fm = pwmEVA(model, θ̂)

    return fm

end
