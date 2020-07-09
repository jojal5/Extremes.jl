abstract type fittedEVA{T<:EVA} end

Base.Broadcast.broadcastable(obj::fittedEVA) = Ref(obj)

"""
    Base.show(io::IO, obj::fittedEVA)

Override of the show function for the objects of type fittedEVA.

"""
function Base.show(io::IO, obj::fittedEVA)

    showfittedEVA(io, obj)

end

include("returnlevel.jl")
include(joinpath("fittedeva", "bayesianeva.jl"))
include(joinpath("fittedeva", "maximumlikelihoodeva.jl"))
include(joinpath("fittedeva", "pwmeva.jl"))
