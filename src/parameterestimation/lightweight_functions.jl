function fit_mle(pd::Type{<:GeneralizedExtremeValue}, y::Vector{<:Real})
    
    fm = gevfit(y)
    fd = Extremes.getdistribution(fm)[]
    
    return fd
end

function fit_pwm(pd::Type{<:GeneralizedExtremeValue}, y::Vector{<:Real})
    
    fm = gevfitpwm(y)
    fd = Extremes.getdistribution(fm)[]
    
    return fd
end

"""
    fit(pd::Type{<:GeneralizedExtremeValue}, y::Vector{<:Real}; method::String="mle")

Fit the Generalized Extreme Value distribution to the data vector `y`.

## Details

This function estimates the parameters of the GEV distribution from an identically distributed random sample. It is a lightweight feature of the Extremes.jl library.

In the presence of non-stationarity or for advanced inference, the [`gevfit`](@ref) function is more appropriate.

The `method` argument allows selecting the estimation method: `mle` for maximum likelihood estimation or `pwm` for the probability-weighted moments method.
"""
function fit(pd::Type{<:GeneralizedExtremeValue}, y::Vector{<:Real}; method::String="mle")
    @assert method ∈ ("mle", "pwm") "Method $method is not a valid method among mle and pwm." 
    
    if method == "mle"
        return fit_mle(pd, y)
    else
        return fit_pwm(pd, y)
    end
    
end

function fit_mle(pd::Type{<:Gumbel}, y::Vector{<:Real})
    
    fm = gumbelfit(y)
    fd = Extremes.getdistribution(fm)[]
    
    return fd
end

function fit_pwm(pd::Type{<:Gumbel}, y::Vector{<:Real})
    
    fm = gumbelfitpwm(y)
    fd = Extremes.getdistribution(fm)[]
    
    return fd
end

"""
    fit(pd::Type{<:Gumbel}, y::Vector{<:Real}; method::String="mle")

Fit the Gumbel distribution to the data vector `y`.

## Details

This function estimates the parameters of the Gumbel distribution from an identically distributed random sample. It is a lightweight feature of the Extremes.jl library.

In the presence of non-stationarity or for advanced inference, the [`gevfit`](@ref) function is more appropriate.

The `method` argument allows selecting the estimation method: `mle` for maximum likelihood estimation or `pwm` for the probability-weighted moments method.
"""
function fit(pd::Type{<:Gumbel}, y::Vector{<:Real}; method::String="mle")
    @assert method ∈ ("mle", "pwm") "Method $method is not a valid method among mle and pwm." 
    
    if method == "mle"
        return fit_mle(pd, y)
    else
        return fit_pwm(pd, y)
    end
    
end

function fit_mle(pd::Type{<:GeneralizedPareto}, y::Vector{<:Real})
    
    fm = gpfit(y)
    fd = Extremes.getdistribution(fm)[]
    
    return fd
end

function fit_pwm(pd::Type{<:GeneralizedPareto}, y::Vector{<:Real})
    
    fm = gpfitpwm(y)
    fd = Extremes.getdistribution(fm)[]
    
    return fd
end

"""
    fit(pd::Type{<:GeneralizedPareto}, y::Vector{<:Real}; method::String="mle")

Fit the Generalized Pareto distribution to the exceedences vector `y`.

## Details

This function estimates the parameters of the GP distribution from an identically distributed random sample. It is a lightweight feature of the Extremes.jl library.

In the presence of non-stationarity or for advanced inference, the [`gpfit`](@ref) function is more appropriate.

The `method` argument allows selecting the estimation method: `mle` for maximum likelihood estimation or `pwm` for the probability-weighted moments method.
"""
function fit(pd::Type{<:GeneralizedPareto}, y::Vector{<:Real}; method::String="mle")
    @assert method ∈ ("mle", "pwm") "Method $method is not a valid method among mle and pwm." 
    
    if method == "mle"
        return fit_mle(pd, y)
    else
        return fit_pwm(pd, y)
    end
    
end