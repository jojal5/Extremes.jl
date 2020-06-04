"""
    gevfitpwm(y::Vector{<:Real})

Estimate the Generalized Extreme value distribution parameters with the
probability weighted moments as described in Hosking et al. (1985).

*Reference:*
Hosking, J. R. M., Wallis, J. R. and Wood, E. F. (1985). Estimation of the generalized extreme-value
    distribution by the method of probability-weighted moments. Technometrics, 27, 251-261.
"""
function gevfitpwm(y::Vector{<:Real})::pwmEVA

    model = BlockMaxima(y)

    fittedmodel = gevfitpwm(model)

    return fittedmodel

end

"""
    gevfitpwm(model::BlockMaxima)

Estimate the Generalized Extreme value distribution parameters with the
probability weighted moments as described in Hosking et al. (1985).

With the methods of moments, it is not possible to include covariates in the
model. If covariates are provided, they are ignored and the stationary model is fitted.

*Reference:*
Hosking, J. R. M., Wallis, J. R. and Wood, E. F. (1985). Estimation of the generalized extreme-value
    distribution by the method of probability-weighted moments. Technometrics, 27, 251-261.
"""
function gevfitpwm(model::BlockMaxima)::pwmEVA

    y = data(model)

    # Computing the estimates of the probability weighted moments M_{1,q,0} for q ∈ {0,1,2}.
    b₀ = pwm(y,1,0,0)
    b₁ = pwm(y,1,1,0)
    b₂ = pwm(y,1,2,0)

    # GEV parameters estimations. Expressions retrieved from Hosking et al. (1985).
    c = (2b₁ - b₀)/(3b₂ - b₀) - log(2)/log(3)
    k = 7.859c + 2.9554c^2
    σ̂ = k *( 2b₁-b₀ ) /(1-2^(-k))/gamma(1+k)
    μ̂ = b₀ - σ̂/k*( 1-gamma(1+k) )

    ξ̂ = -k

    θ̂ = [μ̂; σ̂; ξ̂]

    fm = pwmEVA(model, θ̂)

    return fm

end



"""
    gpfitpwm(y::Vector{<:Real}; threshold::Vector{<:Real}=[0], nobsperblock::Int=1)

Estimate the Generalized Pareto distribution parameters with the probability weighted moments as described
in Hosking & Wallis (1987).

*Reference:*
Hosking, J. R. M. and Wallis, J. R. (1987). Parameter and Quantile Estimation for the Generalized Pareto Distribution,
    Technometrics, 29(3), 339-349.
"""
function gpfitpwm(y::Vector{<:Real}, nobservation::Int ; threshold::Vector{<:Real}=[0], nobsperblock::Int=1)::pwmEVA

    model = PeaksOverThreshold(y, nobservation, threshold = threshold, nobsperblock = nobsperblock)

    fittedmodel = gpfitpwm(model)

    return fittedmodel

end

"""
    gpfitpwm(model::PeaksOverThreshold)

Estimate the Generalized Pareto distribution parameters with the probability weighted moments as described
in Hosking & Wallis (1987).

*Reference:*
Hosking, J. R. M. and Wallis, J. R. (1987). Parameter and Quantile Estimation for the Generalized Pareto Distribution,
    Technometrics, 29(3), 339-349.
"""
function gpfitpwm(model::PeaksOverThreshold)::pwmEVA

#TO DO warn if nonstationary
    y = data(model)

    a₀ = pwm(y,1,0,0)
    a₁ = pwm(y,1,0,1)

    σ̂ = 2*a₀*a₁/(a₀ - 2*a₁)

    k = a₀ / (a₀ - 2*a₁) - 2

    ξ̂ = - k

    θ̂ = [σ̂; ξ̂]

    fm = pwmEVA(model, θ̂)

    return fm

end

"""
    gumbelfitpwm(y::Vector{<:Real})

Estimate the Gumbel distribution with the probability weighted moments as described in Landwehr et al. (1979).

*Reference:*
Landwehr, J. M., Matalas, N. C. and Wallis, J. R. (1979). Probability weighted moments compared with
    some traditional techniques in estimating Gumbel Parameters and quantiles. Water Resources Research,
    15(5), 1055–1064.
"""
function gumbelfitpwm(y::Vector{<:Real})::pwmEVA

    model = BlockMaxima(y)

    fittedmodel = gumbelfitpwm(model)

    return fittedmodel

end


"""
    gumbelfitpwm(model::BlockMaxima)

Estimate the Gumbel distribution with the probability weighted moments as described in Landwehr et al. (1979).

*Reference:*
Landwehr, J. M., Matalas, N. C. and Wallis, J. R. (1979). Probability weighted moments compared with
    some traditional techniques in estimating Gumbel Parameters and quantiles. Water Resources Research,
    15(5), 1055–1064.
"""
function gumbelfitpwm(model::BlockMaxima)::pwmEVA

    y = data(model)

    a₀ = pwm(y,1,0,0)
    a₁ = pwm(y,1,0,1)

    σ̂ = (a₀ - 2a₁)/log(2)

    μ̂ = a₀ - Base.MathConstants.eulergamma*σ̂

    θ̂ = [μ̂, σ̂, 0.0]

    fm = pwmEVA(model,θ̂)

    return fm

end
