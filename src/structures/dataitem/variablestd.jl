
struct VariableStd <: DataItem
    name :: String
    value :: Vector{<:Real}
    offset :: Real
    scale :: Real
end
