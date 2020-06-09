struct pwmEVA{T<:EVA} <: fittedEVA
    "Extreme value model definition"
    model::T
    "Maximum likelihood estimate"
    θ̂::Vector{Float64}
end

"""
    Base.show(io::IO, obj::pwmEVA)

Override of the show function for the objects of type pwmEVA.

"""
function Base.show(io::IO, obj::pwmEVA)

    println(io, "pwmEVA")
    println("model :")
    showEVA(io, obj.model, prefix = "\t")
    println()
    println(io, "θ̂  :\t", obj.θ̂)

end
