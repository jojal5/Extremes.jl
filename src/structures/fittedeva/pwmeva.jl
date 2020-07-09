struct pwmEVA{T, U<:Distribution} <: fittedEVA{T}
    "Extreme value model definition"
    model::T
    "Maximum likelihood estimate"
    θ̂::Vector{Float64}
end

"""
    getdistribution(fittedmodel::pwmEVA)::Vector{<:Distribution}

Return the fitted distribution for the model fitted with the probability weigthed moments.

"""
function getdistribution(fittedmodel::pwmEVA)::Vector{<:Distribution}

    model = fittedmodel.model
    θ̂ = fittedmodel.θ̂

    res = getdistribution(model, θ̂)

    return res

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
    cint(fm::pwmEVA, clevel::Real=.95, nboot::Int=1000)::Array{Array{Float64,1},1}

Estimate the parameter estimates confidence interval by bootstrap.
"""
function cint(fm::pwmEVA, clevel::Real=.95, nboot::Int=5000)::Array{Array{Float64,1},1}

    @assert 0<clevel<1 "the confidence level should be between 0 and 1."
    @assert nboot>0 "the number of bootstrap samples should be positive."

    α = 1-clevel

    y = fm.model.data.value
    n = length(y)

    θ̂ = Array{Float64}(undef, nboot, length(fm.θ̂))

    fitfun = fitpwmfunction(fm)

    for i=1:nboot
        ind = rand(1:n, n)            # Generate a bootstrap sample
        θ̂[i,:] = fitfun(y[ind]).θ̂   # Compute the parameter estimates
    end

    confint = Vector{Vector{Float64}}()

    for c in eachcol(θ̂)
        push!(confint, quantile(c,[α/2; 1-α/2]))
    end

    return confint

end



"""
    fitpwmfunction(fm::pwmEVA{BlockMaxima, GeneralizedExtremeValue})::Function

Returns the corresponding fitpwm function.

"""
function fitpwmfunction(fm::pwmEVA{BlockMaxima, GeneralizedExtremeValue})::Function
    return gevfitpwm
end

"""
    fitpwmfunction(fm::pwmEVA{ThresholdExceedance, GeneralizedPareto})::Function

Returns the corresponding fitpwm function.

"""
function fitpwmfunction(fm::pwmEVA{ThresholdExceedance, GeneralizedPareto})::Function
    return gpfitpwm
end

"""
    fitpwmfunction(fm::pwmEVA{BlockMaxima, Gumbel})::Function

Returns the corresponding fitpwm function.

"""
function fitpwmfunction(fm::pwmEVA{BlockMaxima, Gumbel})::Function
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
    returnlevel(fm::pwmEVA{BlockMaxima, T} where T<:Distribution, returnPeriod::Real)::ReturnLevel

Compute the confidence intervel for the return level corresponding to the return period
`returnPeriod` from the fitted model `fm` with confidence level `confidencelevel`.

"""
function returnlevel(fm::pwmEVA{BlockMaxima, T} where T<:Distribution, returnPeriod::Real)::ReturnLevel

      @assert returnPeriod > zero(returnPeriod) "the return period should be positive."

      # quantile level
      p = 1-1/returnPeriod

      return ReturnLevel(BlockMaximaModel(fm), returnPeriod, quantile(fm, p))

end

"""
    cint(rl::ReturnLevel{pwmEVA{BlockMaxima, T}} where T<:Distribution, confidencelevel::Real=.95)::Vector{Vector{Real}}

Compute the return level corresponding to the return period `returnPeriod` from the fitted model `fm`.

"""
function cint(rl::ReturnLevel{pwmEVA{BlockMaxima, T}} where T<:Distribution, confidencelevel::Real=.95)::Vector{Vector{Real}}

      @assert rl.returnperiod > zero(rl.returnperiod) "the return period should be positive."
      @assert zero(confidencelevel)<confidencelevel<one(confidencelevel) "the confidence level should be in (0,1)."

      # quantile level
      p = 1-1/rl.returnperiod

      # Compute the credible interval

      nboot = 5000
      α = (1 - confidencelevel)

      y = rl.model.fm.model.data.value
      n = length(y)

      qboot = Array{Float64}(undef, nboot)

      fitfun = Extremes.fitpwmfunction(rl.model.fm)

      for i=1:nboot
          ind = rand(1:n, n)            # Generate a bootstrap sample
          θ̂ = fitfun(y[ind]).θ̂          # Compute the parameter estimates
          qboot[i] = quantile(rl.model.fm.model, θ̂, p)[]
      end

      return [quantile(qboot,[α/2, 1-α/2])]

end

"""
    returnlevel(fm::pwmEVA{ThresholdExceedance, T} where T<:Distribution, threshold::Real, nobservation::Int,
        nobsperblock::Int, returnPeriod::Real)::ReturnLevel

Compute the return level corresponding to the return period `returnPeriod` from the fitted model `fm`.

The threshold should be a scalar. A varying threshold is not yet implemented.

"""
function returnlevel(fm::pwmEVA{ThresholdExceedance, T} where T<:Distribution, threshold::Real, nobservation::Int,
    nobsperblock::Int, returnPeriod::Real)::ReturnLevel

    @assert returnPeriod > zero(returnPeriod) "the return period should be positive."

    # Exceedance probability
    ζ = length(fm.model.data.value)/nobservation

    # Appropriate quantile level given the probability exceedance and the number of obs per year
    p = 1-1/(returnPeriod * nobsperblock * ζ)

    return ReturnLevel(PeakOverThreshold(fm, threshold, nobservation, nobsperblock),
        returnPeriod, threshold .+ quantile(fm, p))

end

"""
cint(rl::ReturnLevel{pwmEVA{ThresholdExceedance, T}} where T<:Distribution, threshold::Real, nobservation::Int,
    nobsperblock::Int, confidencelevel::Real=.95)::Vector{Vector{Real}}

Compute the confidence intervel for the return level corresponding to the return period
`returnPeriod` from the fitted model `fm` with confidence level `confidencelevel`.

The threshold should be a scalar. A varying threshold is not yet implemented.

"""
function cint(rl::ReturnLevel{pwmEVA{ThresholdExceedance, T}} where T<:Distribution, confidencelevel::Real=.95)::Vector{Vector{Real}}

    @assert rl.returnperiod > zero(rl.returnperiod) "the return period should be positive."
    @assert zero(confidencelevel)<confidencelevel<one(confidencelevel) "the confidence level should be in (0,1)."

    # Exceedance probability
    ζ = length(rl.model.fm.model.data.value)/rl.model.nobservation

    # Appropriate quantile level given the probability exceedance and the number of obs per year
    p = 1-1/(rl.returnperiod * rl.model.nobsperblock * ζ)

    # Compute the credible interval

    nboot = 5000
    α = (1 - confidencelevel)

    y = rl.model.fm.model.data.value
    n = length(y)

    qboot = Array{Float64}(undef, nboot)

    fitfun = Extremes.fitpwmfunction(rl.model.fm)

    for i=1:nboot
        ind = rand(1:n, n)            # Generate a bootstrap sample
        θ̂ = fitfun(y[ind]).θ̂          # Compute the parameter estimates
        qboot[i] = quantile(rl.model.fm.model, θ̂, p)[]
    end

    return [rl.model.threshold .+ quantile(qboot,[α/2, 1-α/2])]

end

"""
    showfittedEVA(io::IO, obj::pwmEVA; prefix::String = "")

Displays a pwmEVA with the prefix `prefix` before every line.

"""
function showfittedEVA(io::IO, obj::pwmEVA; prefix::String = "")

    println(io, prefix, "pwmEVA")
    println(io, prefix, "model :")
    showEVA(io, obj.model, prefix = prefix*"\t")
    println(io)
    println(io, prefix, "θ̂  :\t", obj.θ̂)

end
