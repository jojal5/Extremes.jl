"""
    gevfitpwm(y::Vector{<:Real})::pwmEVA

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
    gevfitpwm(model::BlockMaxima)::pwmEVA

Estimate the Generalized Extreme value distribution parameters with the
probability weighted moments as described in Hosking et al. (1985).

With the methods of moments, it is not possible to include covariates in the
model. If covariates are provided, they are ignored and the stationary model is fitted.

*Reference:*
Hosking, J. R. M., Wallis, J. R. and Wood, E. F. (1985). Estimation of the generalized extreme-value
    distribution by the method of probability-weighted moments. Technometrics, 27, 251-261.

"""
function gevfitpwm(model::BlockMaxima{GeneralizedExtremeValue})::pwmEVA

    model = validatestationarity(model)

    y = model.data

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
    ϕ̂ = log(σ̂)

    θ̂ = [μ̂; ϕ̂; ξ̂]

    fm = pwmEVA(model, θ̂)

    return fm

end
