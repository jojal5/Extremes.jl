"""
    gumbelfitpwm(y::Vector{<:Real})::pwmEVA

Estimate the Gumbel distribution with the probability weighted moments as described in Landwehr et al. (1979).

*Reference:*
Landwehr, J. M., Matalas, N. C. and Wallis, J. R. (1979). Probability weighted moments compared with
    some traditional techniques in estimating Gumbel Parameters and quantiles. Water Resources Research,
    15(5), 1055–1064.

"""
function gumbelfitpwm(y::Vector{<:Real})::pwmEVA

    model = BlockMaxima(Variable("y", y))

    fittedmodel = gumbelfitpwm(model)

    return fittedmodel

end

"""
    gumbelfitpwm(df::DataFrame, datacol::Symbol)::pwmEVA

Estimate the Gumbel distribution with the probability weighted moments as described in Landwehr et al. (1979).

*Reference:*
Landwehr, J. M., Matalas, N. C. and Wallis, J. R. (1979). Probability weighted moments compared with
    some traditional techniques in estimating Gumbel Parameters and quantiles. Water Resources Research,
    15(5), 1055–1064.

"""
function gumbelfitpwm(df::DataFrame, datacol::Symbol)::pwmEVA

    model = BlockMaxima(Variable(string(datacol), df[:, datacol]))

    fittedmodel = gumbelfitpwm(model)

    return fittedmodel

end

"""
    gumbelfitpwm(model::BlockMaxima)::pwmEVA

Estimate the Gumbel distribution with the probability weighted moments as described in Landwehr et al. (1979).

*Reference:*
Landwehr, J. M., Matalas, N. C. and Wallis, J. R. (1979). Probability weighted moments compared with
    some traditional techniques in estimating Gumbel Parameters and quantiles. Water Resources Research,
    15(5), 1055–1064.

"""
function gumbelfitpwm(model::BlockMaxima)::pwmEVA

    model = validatestationarity(model)

    y = model.data.value

    a₀ = pwm(y,1,0,0)
    a₁ = pwm(y,1,0,1)

    σ̂ = (a₀ - 2a₁)/log(2)

    μ̂ = a₀ - Base.MathConstants.eulergamma*σ̂

    ϕ̂ = log(σ̂)

    θ̂ = [μ̂, ϕ̂, 0.0]

    fm = pwmEVA{BlockMaxima, Gumbel}(model,θ̂)

    return fm

end
