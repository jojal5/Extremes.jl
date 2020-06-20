
struct Cluster
    "first threshold"
    u₁::Real
    "second threshold"
    u₂::Real
    "positions of clustered exceedences"
    position::Vector{<:Int}
    "values of clustered exceedences"
    value::Vector{<:Real}

    function Cluster(u₁::Real,u₂::Real,position::Vector{<:Int},value::Vector{<:Real})
        @assert u₁ >= u₂ "the second threshold should not be higher than the first one."
        @assert length(position) == length(value) "the vectors of the clustered exceedances positions and the values should be of the same length."

        return new(u₁,u₂,position,value)
    end
end



Base.Broadcast.broadcastable(obj::Cluster) = Ref(obj)

"""
    Base.show(io::IO, obj::Cluster)

Override of the show function for the objects of type EVA.

"""
function Base.show(io::IO, obj::Cluster)

    showCluster(io, obj)

end

"""
    showCluster(io::IO, obj::Cluster; prefix::String = " ")

Displays a Cluster with the prefix `prefix` before every line.

"""
function showCluster(io::IO, obj::Cluster; prefix::String = "")

    println(io, prefix, "Cluster")
    println(io, prefix, "u₁: ", obj.u₁)
    println(io, prefix, "u₂: ", obj.u₂)
    println(io, prefix, "position: ", obj.position)
    println(io, prefix, "value: ", obj.value)

end
