
struct Variable <: DataItem
    name :: String
    value :: Vector{<:Real}
end

"""
    standardize(v::Variable)::VariableStd

Standardize the vector of values in `v.values` and return a type `VariableStd`.
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
