

abstract type ReturnLevelModel{T<:AbstractFittedExtremeValueModel} end

"""
    ReturnLevel

ReturnLevel type constructed by the function [`returnlevel`](@ref).
"""
struct ReturnLevel{T<:AbstractFittedExtremeValueModel}
      model::ReturnLevelModel{T}
      returnperiod::Real
      value::Array{<:Real}
end

# struct BlockMaximaModel{T<:AbstractFittedExtremeValueModel{BlockMaxima}} <: ReturnLevelModel{T}
#   fm::T
# end

struct BlockMaximaModel{obj<:AbstractFittedExtremeValueModel{BlockMaxima{T}} where T} <: ReturnLevelModel{obj}
  fm::obj
end

struct PeakOverThreshold{T<:AbstractFittedExtremeValueModel{ThresholdExceedance}} <: ReturnLevelModel{T}
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
