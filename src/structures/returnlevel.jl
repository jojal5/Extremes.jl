abstract type ReturnLevelModel end

struct ReturnLevel{T<:ReturnLevelModel}
      model::T
      returnperiod::Real
      value::Vector{<:Real}
end

struct BlockMaximaModel{T<:fittedEVA{BlockMaxima}} <: ReturnLevelModel
  fm::T
end

struct PeakOverThreshold{T<:fittedEVA{ThresholdExceedance}} <: ReturnLevelModel
  fm::T
  threshold::Real
  nobservation::Int
  nobsperblock::Int
end

"""
    Base.show(io::IO, obj::ReturnLevel)

Override of the show function for the objects of type ReturnLevel.
"""
function Base.show(io::IO, obj::ReturnLevel)

    println(io, "ReturnLevel")
    println(io, "returnperiod :\t", obj.returnperiod)
    println(io, "value :\t\t", typeof(obj.value), "[", length(obj.value), "]")

end
