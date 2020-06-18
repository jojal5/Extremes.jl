
struct VariableStd <: DataItem
    name :: String
    value :: Vector{<:Real}
    offset :: Real
    scale :: Real
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
