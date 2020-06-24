
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
    getcluster(y::Vector{<:Real}, u::Real; runlength::Int=1)::Vector{Cluster}

Threshold exceedances separated by fewer than *r* non-exceedances belong to the same cluster. The value *r* is corresponds to the runlength parameter.
This approach is referred to as the *runs declustering scheme* (see Coles, 2001 sec. 5.3.2).
"""
function getcluster(y::Vector{<:Real}, u::Real; runlength::Int=1)

    cluster = getcluster(y, u, u)

    if length(cluster)>1
        runscluster = Vector{Cluster}()
        cluster_current = cluster[1]

        for i=2:length(cluster)

            s = cluster[i].position[1] - cluster_current.position[end] - 1

            if s < runlength
                cluster_current = Extremes.merge(cluster_current, cluster[i])
            else
                push!(runscluster, cluster_current)
                cluster_current = cluster[i]
            end
        end
        push!(runscluster, cluster_current)

        return runscluster
    else
        return cluster
    end

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
    merge(c₁::Cluster, c₂::Cluster)

Merge cluster c₁ and c₂ into a single cluster.
"""
function merge(c₁::Cluster, c₂::Cluster)

    @assert c₁.u₁ ≈ c₂.u₁ "both clusters should have the same threshold to be merged."
    @assert c₁.u₂ ≈ c₂.u₂ "both clusters should have the same threshold to be merged."

    position = vcat(c₁.position, c₂.position)
    value = vcat(c₁.value, c₂.value)

    c = Cluster(c₁.u₁, c₁.u₂, position, value)

    return c

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
