
struct VariableStd <: DataItem
    name :: String
    value :: Vector{<:Real}
    offset :: Real
    scale :: Real
end

"""
    VariableStd(name::String, z::Vector{<:Real})::VariableStd

Construct a VariableStd type from the standardized vector `z` with the name `name`.
"""
function VariableStd(name::String, z::Vector{<:Real})::VariableStd

    m = mean(z)
    s = std(z)

    @assert isapprox(m, 0.0, atol = sqrt(eps())) "the mean should be equal to zero. Use the type Variable instead."
    @assert isapprox(s, 1.0, atol = sqrt(eps())) "the standard deviation should be equal to one. Use the type Variable instead."

    return VariableStd(name,z,0,1)
end

"""
    reconstruct(vstd::VariableStd)::Variable

Reconstruct the orginial variable from the standardized one.
"""
function reconstruct(vstd::VariableStd)::Variable

    z = vstd.value

    x = vstd.scale*z .+ vstd.offset

    v = Variable(vstd.name, x)

    return v

end
