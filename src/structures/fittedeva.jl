abstract type fittedEVA end

struct ReturnLevel
      fittedmodel::fittedEVA
      returnperiod::Real
      value::Vector{<:Real}
      cint::Vector{Vector{T}} where T<:Real
end

Base.Broadcast.broadcastable(obj::fittedEVA) = Ref(obj)

"""
    Base.show(io::IO, obj::fittedEVA)

Override of the show function for the objects of type fittedEVA.

"""
function Base.show(io::IO, obj::fittedEVA)

    showfittedEVA(io, obj)

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


include(joinpath("fittedeva", "bayesianeva.jl"))
include(joinpath("fittedeva", "maximumlikelihoodeva.jl"))
include(joinpath("fittedeva", "pwmeva.jl"))
