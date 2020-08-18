
abstract type DataItem end

Base.Broadcast.broadcastable(obj::DataItem) = Ref(obj)

include(joinpath("dataitem", "variable.jl"))
include(joinpath("dataitem", "variablestd.jl"))
