
"""
    gumbelfitpwm(...)

Estimate the Gumbel parameters with the probability weighted moments.

# Implementation

Estimation with the probability weighted moments, as described by [Landwehr *et al. (1979)](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/WR015i005p01055),
is only possible in the stationary case.

See also [`gumbelfitpwm`](@ref) for the other methods, [`gevfit`](@ref), [`gevfitbayes`](@ref) and [`BlockMaxima`](@ref).

# Reference

Landwehr, J. M., Matalas, N. C. and Wallis, J. R. (1979). Probability weighted moments compared with
    some traditional techniques in estimating Gumbel parameters and quantiles. *Water Resources Research*,
    15:1055–1064.
"""
function gumbelfitpwm end



"""
    gumbelfitpwm(y::Vector{<:Real})

Estimate the Gumbel parameters with the probability weighted moments.

"""
function gumbelfitpwm(y::Vector{<:Real})::pwmEVA

    model = BlockMaxima{Gumbel}(Variable("y", y))

    fittedmodel = gumbelfitpwm(model)

    return fittedmodel

end

"""
    gumbelfitpwm(df::DataFrame, datacol::Symbol)::pwmEVA

Estimate the Gumbel parameters with the probability weighted moments.

Block maxima data are in the column `datacol` of the dataframe `df`.
"""
function gumbelfitpwm(df::DataFrame, datacol::Symbol)::pwmEVA

    model = BlockMaxima{Gumbel}(Variable(string(datacol), df[:, datacol]))

    fittedmodel = gumbelfitpwm(model)

    return fittedmodel

end

"""
    gumbelfitpwm(model::BlockMaxima{Gumbel})

Estimate the Gumbel parameters with the probability weighted moments.
"""
function gumbelfitpwm(model::BlockMaxima{Gumbel})::pwmEVA

    model = validatestationarity(model)

    y = model.data.value

    a₀ = pwm(y,1,0,0)
    a₁ = pwm(y,1,0,1)

    σ̂ = (a₀ - 2a₁)/log(2)

    μ̂ = a₀ - Base.MathConstants.eulergamma*σ̂

    ϕ̂ = log(σ̂)

    θ̂ = [μ̂, ϕ̂, 0.0]

    fm = pwmEVA{BlockMaxima{Gumbel}}(model,θ̂)

    return fm

end
