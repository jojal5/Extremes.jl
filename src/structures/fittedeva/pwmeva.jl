struct pwmEVA{T<:EVA} <: fittedEVA
    "Extreme value model definition"
    model::T
    "Maximum likelihood estimate"
    θ̂::Vector{Float64}
end


"""
    quantile(fm::pwmEVA, p::Real)::Vector{<:Real}

Compute the quantile of level `p` from the fitted model. For the probability weighted moment method, the model has to be stationary.

"""
function quantile(fm::pwmEVA, p::Real)::Vector{<:Real}

    @assert zero(p)<p<one(p) "the quantile level should be between 0 and 1."

    q = quantile(fm.model, fm.θ̂, p)

    return q

end

"""
    parametervar(fm::pwmEVA, nboot::Int=1000)

Estimate the parameter estimates covariance matrix by bootstrap.
"""
function parametervar(fm::pwmEVA, nboot::Int=1000)::Array{Float64, 2}

    @assert nboot>0 "the number of bootstrap samples should be positive."

    y = fm.model.data.value
    n = length(y)

    θ̂ = Array{Float64}(undef, nboot, length(fm.θ̂))

    fitfun = fitpwmfunction(fm)

    for i=1:nboot
        ind = rand(1:n, n)            # Generate a bootstrap sample
        θ̂[i,:] = fitfun(y[ind]).θ̂   # Compute the parameter estimates
    end

    V = cov(θ̂)                    # Compute the approximate covariance matrix

    return V

end

"""
    fitpwmfunction(fm::pwmEVA{BlockMaxima{GeneralizedExtremeValue}})::Function

Returns the corresponding fitpwm function.

"""
function fitpwmfunction(fm::pwmEVA{BlockMaxima{GeneralizedExtremeValue}})::Function
    return gevfitpwm
end

"""
    fitpwmfunction(fm::pwmEVA{BlockMaxima{GeneralizedPareto}})::Function

Returns the corresponding fitpwm function.

"""
function fitpwmfunction(fm::pwmEVA{ThresholdExceedance})::Function
    return gpfitpwm
end

"""
    fitpwmfunction(fm::pwmEVA{BlockMaxima{Gumbel}})::Function

Returns the corresponding fitpwm function.

"""
function fitpwmfunction(fm::pwmEVA{BlockMaxima{Gumbel}})::Function
    return gumbelfitpwm
end

"""
    quantilevar(fm::pwmEVA, level::Real, nboot::Int=1000)::Vector{<:Real}

Compute the  approximate variance of the quantile of level `level` from the fitted model `fm` by bootstrap.

"""
function quantilevar(fm::pwmEVA, level::Real, nboot::Int=1000)::Vector{<:Real}

    # Compute the approximate covariance matrice of the parameter estimates by bootstrap
    V = parametervar(fm, nboot)

    # Compute the approximate quantile variance by the delta method
    f(θ::DenseVector) = quantile(fm.model,θ,level)[]  # With the pwm method, the model is stationary
    Δf(θ::DenseVector) = ForwardDiff.gradient(f, θ)
    G = Δf(fm.θ̂)

    qv = G'*V*G

    return [qv]

end


"""
    Base.show(io::IO, obj::pwmEVA)

Override of the show function for the objects of type pwmEVA.

"""
function Base.show(io::IO, obj::pwmEVA)

    println(io, "pwmEVA")
    println(io, "model :")
    showEVA(io, obj.model, prefix = "\t")
    println(io)
    println(io, "θ̂  :\t", obj.θ̂)

end
