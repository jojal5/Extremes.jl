
struct ReturnLevel
      fittedmodel::fittedEVA
      returnperiod::Real
      value::Vector{<:Real}
      cint::Vector{Vector{T}} where T<:Real
end

"""
    returnlevel(fm::BayesianEVA{BlockMaxima}, returnPeriod::Real, confidencelevel::Real=.95)::ReturnLevel

Compute the return level of the return period `returnPeriod` from the fitted model `fm`.

"""
function returnlevel(fm::BayesianEVA{BlockMaxima{GeneralizedExtremeValue}}, returnPeriod::Real, confidencelevel::Real=.95)::ReturnLevel

      @assert returnPeriod > zero(returnPeriod) "the return period should be positive."
      @assert zero(confidencelevel)<confidencelevel<one(confidencelevel) "the confidence level should be in (0,1)."

      α = (1 - confidencelevel)

      # quantile level
      p = 1-1/returnPeriod

      Q = quantile(fm, p)

      q = vec(mean(Q, dims=1))

      qsliced = slicematrix(Q)

      a = quantile.(qsliced, α/2)
      b = quantile.(qsliced, 1-α/2)

      cint = slicematrix(hcat(a,b), dims=2)

      res = ReturnLevel(fm, returnPeriod, q, cint)

      return res

end

"""
    returnlevel(fm::BayesianEVA{ThresholdExceedance}, threshold::Vector{<:Real}, nobservation::Int,
        nobsperblock::Int, returnPeriod::Real, confidencelevel::Real=.95)::ReturnLevel

Compute the return level of the return period `returnPeriod` from the fitted model `fm`.

"""
function returnlevel(fm::BayesianEVA{ThresholdExceedance}, threshold::Vector{<:Real}, nobservation::Int,
    nobsperblock::Int, returnPeriod::Real, confidencelevel::Real=.95)::ReturnLevel

    @assert returnPeriod > zero(returnPeriod) "the return period should be positive."
    @assert zero(confidencelevel)<confidencelevel<one(confidencelevel) "the confidence level should be in (0,1)."

    α = (1 - confidencelevel)

    # Exceedance probability
    ζ = length(fm.model.data.value)/nobservation

    # Appropriate quantile level given the probability exceedance and the number of obs per year
    p = 1-1/(returnPeriod * nobsperblock * ζ)

    Q = quantile(fm, p)

    q = threshold .+ vec(mean(Q, dims=1))

    qsliced = slicematrix(Q)

    a = threshold .+ quantile.(qsliced, α/2)
    b = threshold .+ quantile.(qsliced, 1-α/2)

    cint = slicematrix(hcat(a,b), dims=2)

    res = ReturnLevel(fm, returnPeriod, q, cint)

    return res

end

"""
    returnlevel(fm::MaximumLikelihoodEVA{BlockMaxima}, returnPeriod::Real, confidencelevel::Real=.95)::ReturnLevel

Compute the return level of the return period `returnPeriod` from the fitted model `fm`.

"""
function returnlevel(fm::MaximumLikelihoodEVA{BlockMaxima}, returnPeriod::Real, confidencelevel::Real=.95)::ReturnLevel

      @assert returnPeriod > zero(returnPeriod) "the return period should be positive."
      @assert zero(confidencelevel)<confidencelevel<one(confidencelevel) "the confidence level should be in (0,1)."

      α = (1 - confidencelevel)

      # quantile level
      p = 1-1/returnPeriod

      q = quantile(fm, p)

      v = quantilevar(fm,p)

      qdist = Normal.(q,sqrt.(v))

      a = quantile.(qdist,α/2)
      b = quantile.(qdist,1-α/2)

      cint = slicematrix(hcat(a,b), dims=2)

      res = ReturnLevel(fm, returnPeriod, q, cint)

      return res

end

"""
    returnlevel(fm::MaximumLikelihoodEVA{ThresholdExceedance}, threshold::Vector{<:Real}, nobservation::Int,
        nobsperblock::Int, returnPeriod::Real, confidencelevel::Real=.95)::ReturnLevel

Compute the return level of the return period `returnPeriod` from the fitted model `fm`.

The threshold should be a scalar. A varying threshold is not yet implemented.

"""
function returnlevel(fm::MaximumLikelihoodEVA{ThresholdExceedance}, threshold::Real, nobservation::Int,
    nobsperblock::Int, returnPeriod::Real, confidencelevel::Real=.95)::ReturnLevel

    @assert returnPeriod > zero(returnPeriod) "the return period should be positive."
    @assert zero(confidencelevel)<confidencelevel<one(confidencelevel) "the confidence level should be in (0,1)."

    α = (1 - confidencelevel)

    # Exceedance probability
    ζ = length(fm.model.data.value)/nobservation

    # Appropriate quantile level given the probability exceedance and the number of obs per year
    p = 1-1/(returnPeriod * nobsperblock * ζ)

    q = threshold .+ quantile(fm, p)

    # Computing the variance corresponding to ζ
    f(θ::Vector{<:Real}) = Extremes.quantile(fm.model,fm.θ̂,1-1/(returnPeriod * nobsperblock * θ[]))[]
    v₁ = (ForwardDiff.gradient(f, [ζ])[])^2*ζ*(1-ζ)/nobservation

    # This component seems to be forgoten by Coles (2001) in Section 4.4.1

    # Computing the variance corresponding to the other parameters
    v₂ = Extremes.quantilevar(fm, p)

    v = v₁ .+ v₂

    qdist = Normal.(q,sqrt.(v))

    a = quantile.(qdist,α/2)
    b = quantile.(qdist,1-α/2)

    cint = Extremes.slicematrix(hcat(a,b), dims=2)

    res = ReturnLevel(fm, returnPeriod, q, cint)

    return res

end


"""
    Base.show(io::IO, obj::ReturnLevel)

Override of the show function for the objects of type ReturnLevel.
"""
function Base.show(io::IO, obj::ReturnLevel)

    println(io, "ReturnLevel")
    println(io, "fittedmodel :")
    showfittedEVA(io, obj.fittedmodel, prefix = "\t\t")
    println(io, "returnperiod :\t", obj.returnperiod)
    println(io, "value :\t\t", typeof(obj.value), "[", length(obj.value), "]")
    println(io, "cint :\t\t", typeof(obj.cint), "[", length(obj.cint), "]")

end
