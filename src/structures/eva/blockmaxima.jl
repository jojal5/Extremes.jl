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

function paramindex(model::BlockMaxima)::Dict{Symbol,Vector{<:Int}}

    sμ = length(model.location.covariate) + 1
    sϕ = length(model.logscale.covariate) + 1
    sξ = length(model.shape.covariate) + 1

    return Dict{Symbol,Vector{<:Int}}(
        :μ => Int64[k for k in 1:sμ],
        :ϕ => Int64[k for k in sμ .+ (1:sϕ)],
        :ξ => Int64[k for k in sμ .+ sϕ .+ (1:sξ)]
    )

end


function getcovariatenumber(model::BlockMaxima)::Int

    return sum([length(model.location.covariate), length(model.logscale.covariate), length(model.shape.covariate)])

end


function nparameter(model::BlockMaxima)::Int

    return 3 + getcovariatenumber(model)

end


function getdistribution(model::BlockMaxima, θ::AbstractVector{<:Real})::Vector{<:Distribution}

    @assert length(θ)==nparameter(model) "The length of the parameter vector should be equal to the model number of parameters."

    pi = paramindex(model)
    μ = model.location.fun(θ[pi[:μ]])
    ϕ = model.logscale.fun(θ[pi[:ϕ]])
    ξ = model.shape.fun(θ[pi[:ξ]])

    σ = exp.(ϕ)

    fd = GeneralizedExtremeValue.(μ, σ, ξ)

    return fd

end


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
