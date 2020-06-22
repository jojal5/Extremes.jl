
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
    getcluster(y::Vector{<:Real}, u₁::Real, u₂::Real)::Vector{Cluster}

Returns a DataFrame with clusters for exceedance models. A cluster is defined as a sequence where values are higher than u₂ with at least a value higher than threshold u₁.

"""
function getcluster(y::Vector{<:Real}, u₁::Real, u₂::Real)::Vector{Cluster}

    @assert u₁ >= u₂ "the second threshold should not be higher than the first one."

    n = length(y)

    exceedancePosition = findall(y .> u₁)

    clusterEnd = 0

    cluster = Vector{Cluster}()

    for i in exceedancePosition

            if i > clusterEnd

               j = 1

                while (i-j) > 0
                    if y[i-j] > u₂
                        j += 1
                    else
                        break
                    end
                end

                k = 1

                while (i+k) < (n+1)
                    if y[i+k] > u₂
                        k += 1
                    else
                        break
                    end
                end

            ind = i-(j-1) : i+(k-1)

            clusterEnd = ind[end]

            c = Cluster(u₁, u₂, collect(ind), y[ind])

            push!(cluster, c)

            end

    end

    return cluster

end

"""
    getcluster(y::Vector{<:Real}, u::Real)::Vector{Cluster}

Returns a DataFrame with clusters for exceedance models. A cluster is defined as a sequence where values are higher than u.

"""
function getcluster(y::Vector{<:Real}, u::Real)::Vector{Cluster}

    return getcluster(y, u, u)

end



"""
    length(c::Cluster)

Compute the cluster length.
"""
function length(c::Cluster)
    return length(c.position)
end

"""
    max(c::Cluster)

Compute the cluster maximum.
"""
function maximum(c::Cluster)
    return maximum(c.value)
end


"""
    sum(c::Cluster)

Compute the cluster sum.
"""
function sum(c::Cluster)
    return sum(c.value)
end


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
