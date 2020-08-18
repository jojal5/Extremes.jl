"""
    Variable(name::String, value :: Vector{<:Real})

Construct a Variable type
"""
struct Variable <: DataItem
    name :: String
    value :: Vector{<:Real}
end

"""
    standardize(v::Variable)

Standardize the values of the Variable.

# Implementation

The Variable values are standardized by substracting the empirical mean
and dividing by the empirical standard deviation. A [`VariableStd`](@ref) type
is returned.

See also [`Variable`](@ref), [`VariableStd`](@ref) and [`reconstruct`](@ref).
"""
function standardize(v::Variable)::VariableStd

    x = v.value

    x̄ = mean(x)
    s = std(x)

    offset = isapprox(x̄,0) ? zero(typeof(x[1])) : x̄
    scale = (isapprox(s,1) || isapprox(s,0.0)) ? one(typeof(x[1])) : s

    z = (x .- offset) ./ scale

    vstd = VariableStd(v.name, z, offset, scale)

    return vstd

end
