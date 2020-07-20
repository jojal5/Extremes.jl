
"""
    gevfitpwm(...)

Estimate the GEV parameters with the probability weighted moments.

# Implementation

Estimation with the probability weighted moments, as described by [Hosking *et al. (1985)](https://www.tandfonline.com/doi/abs/10.1080/00401706.1985.10488049),
is only possible in the stationary case.

See also [`gevfitpwm`](@ref) for the other methods, [`gevfit`](@ref), [`gevfitbayes`](@ref) and [`BlockMaxima`](@ref).

# Reference

Hosking, J. R. M., Wallis, J. R. and Wood, E. F. (1985). Estimation of the generalized extreme-value
    distribution by the method of probability-weighted moments. *Technometrics*, 27:251-261.
"""
function gevfitpwm end

"""
    gevfitpwm(y::Vector{<:Real})

Estimate the GEV parameters with the probability weighted moments.

"""
function gevfitpwm(y::Vector{<:Real})::pwmEVA

    model = BlockMaxima(Variable("y", y))

    fittedmodel = gevfitpwm(model)

    return fittedmodel

end

"""
    gevfitpwm(df::DataFrame, datacol::Symbol)

Estimate the GEV parameters with the probability weighted moments.

Block maxima data are in the column `datacol` of the dataframe `df`.

"""
function gevfitpwm(df::DataFrame, datacol::Symbol)::pwmEVA

    model = BlockMaxima(Variable(string(datacol), df[:, datacol]))

    fittedmodel = gevfitpwm(model)

    return fittedmodel

end

"""
    gevfitpwm(model::BlockMaxima)

Estimate the GEV parameters with the probability weighted moments.

"""
function gevfitpwm(model::BlockMaxima)::pwmEVA

    model = validatestationarity(model)

    y = model.data.value

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

    fm = pwmEVA{BlockMaxima, GeneralizedExtremeValue}(model, θ̂)

    return fm

end
