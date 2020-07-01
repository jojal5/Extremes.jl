struct BlockMaxima <: EVA
    data::Variable
    location::paramfun
    logscale::paramfun
    shape::paramfun
end

"""
    BlockMaxima(data::Vector{<:Real};
        locationcov::Vector{Variable} = Vector{Variable}(),
        logscalecov::Vector{Variable} = Vector{Variable}(),
        shapecov::Vector{Variable} = Vector{Variable}())::BlockMaxima

Creates a BlockMaxima structure.

"""
function BlockMaxima(data::Variable;
    locationcov::Vector{<:DataItem} = Vector{Variable}(),
    logscalecov::Vector{<:DataItem} = Vector{Variable}(),
    shapecov::Vector{<:DataItem} = Vector{Variable}())::BlockMaxima

    n = length(data.value)
    validatelength(n, locationcov)
    validatelength(n, logscalecov)
    validatelength(n, shapecov)

    locationfun = computeparamfunction(locationcov)
    logscalefun = computeparamfunction(logscalecov)
    shapefun = computeparamfunction(shapecov)

    return BlockMaxima(data, paramfun(locationcov, locationfun), paramfun(logscalecov, logscalefun), paramfun(shapecov, shapefun))

end

"""
    paramindex(model::BlockMaxima)::Dict{Symbol,Vector{<:Int}}

Get the parameter indexing for a BlockMaxima.

"""
function paramindex(model::BlockMaxima)::Dict{Symbol,Vector{<:Int}}

    i = 0
    function increasei()
        i = i + 1
        return i
    end
    return Dict{Symbol,Vector{<:Int}}(
        :μ => Int64[increasei() for k in 1:length(model.location.covariate) + 1],
        :ϕ => Int64[increasei() for k in 1:length(model.logscale.covariate) + 1],
        :ξ => Int64[increasei() for k in 1:length(model.shape.covariate) + 1]
    )

end

"""
    getcovariatenumber(model::BlockMaxima)::Int

Return the number of covariates.

"""
function getcovariatenumber(model::BlockMaxima)::Int

    return sum([length(model.location.covariate), length(model.logscale.covariate), length(model.shape.covariate)])

end

"""
    nparameter(model::BlockMaxima)::Int

Get the number of parameters in a BlockMaxima.

"""
function nparameter(model::BlockMaxima)::Int

    return 3 + getcovariatenumber(model)

end

"""
    getdistribution(model::BlockMaxima, θ::Vector{<:Real})::Vector{<:Distribution}

Return the fitted distribution in case of stationarity or the vector of fitted distribution in case of non-stationarity.

"""
function getdistribution(model::BlockMaxima, θ::Vector{<:Real})::Vector{<:Distribution}

    @assert length(θ)==nparameter(model) "The length of the parameter vector should be equal to the model number of parameters."

    pi = paramindex(model)
    μ = model.location.fun(θ[pi[:μ]])
    ϕ = model.logscale.fun(θ[pi[:ϕ]])
    ξ = model.shape.fun(θ[pi[:ξ]])

    σ = exp.(ϕ)

    fd = GeneralizedExtremeValue.(μ, σ, ξ)

    return fd

end

"""
    getinitialvalue(model::BlockMaxima)::Vector{<:Real}

Get an initial values vector for the parameters of model.

"""
function getinitialvalue(model::BlockMaxima)::Vector{<:Real}

    y = model.data.value

    # Compute stationary initial values
    μ₀,ϕ₀,ξ₀ = getinitialvalue(GeneralizedExtremeValue,y)
    # Store them in a dictionary
    θ₀ = Dict(:μ => μ₀, :ϕ => ϕ₀, :ξ => ξ₀)

    initialvalues = zeros(nparameter(model))
    pi = paramindex(model)
    for param in [:μ, :ϕ, :ξ]
        ind = pi[param][1]
        initialvalues[ind] = θ₀[param]
    end

    return initialvalues

end

"""
    showEVA(io::IO, obj::BlockMaxima; prefix::String = "")

Displays a BlockMaxima with the prefix `prefix` before every line.

"""
function showEVA(io::IO, obj::BlockMaxima; prefix::String = "")

    println(io, prefix, "BlockMaxima")
    println(io, prefix, "data :\t\t", typeof(obj.data.value), "[", length(obj.data.value), "]")
    println(io, prefix, "location :\t", showparamfun("μ", obj.location))
    println(io, prefix, "logscale :\t", showparamfun("ϕ", obj.logscale))
    println(io, prefix, "shape :\t\t", showparamfun("ξ", obj.shape))

end
