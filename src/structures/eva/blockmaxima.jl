struct BlockMaxima{T<:Distribution} <: EVA
    data::Vector{<:Real}
    location::paramfun
    logscale::paramfun
    shape::paramfun
end

"""
    BlockMaxima(data::Vector{<:Real};
        locationcov::Vector{ExplanatoryVariable} = Vector{ExplanatoryVariable}(),
        logscalecov::Vector{ExplanatoryVariable} = Vector{ExplanatoryVariable}(),
        shapecov::Vector{ExplanatoryVariable} = Vector{ExplanatoryVariable}())::BlockMaxima

Creates a BlockMaxima structure.

"""
function BlockMaxima(data::Vector{<:Real};
    locationcov::Vector{ExplanatoryVariable} = Vector{ExplanatoryVariable}(),
    logscalecov::Vector{ExplanatoryVariable} = Vector{ExplanatoryVariable}(),
    shapecov::Vector{ExplanatoryVariable} = Vector{ExplanatoryVariable}(),
    dist::Type{<:Distribution} = GeneralizedExtremeValue)::BlockMaxima

    n = length(data)
    validatelength(n, locationcov)
    validatelength(n, logscalecov)
    validatelength(n, shapecov)

    locationfun = computeparamfunction(locationcov)
    logscalefun = computeparamfunction(logscalecov)
    shapefun = computeparamfunction(shapecov)

    return BlockMaxima{dist}(data, paramfun(locationcov, locationfun), paramfun(logscalecov, logscalefun), paramfun(shapecov, shapefun))

end

"""
    BlockMaxima{T}(data::Vector{<:Real};
        locationcov::Vector{ExplanatoryVariable} = Vector{ExplanatoryVariable}(),
        logscalecov::Vector{ExplanatoryVariable} = Vector{ExplanatoryVariable}(),
        shapecov::Vector{ExplanatoryVariable} = Vector{ExplanatoryVariable}())::BlockMaxima where T<:Distribution

Creates a BlockMaxima structure.

"""
function BlockMaxima{T}(data::Vector{<:Real};
    locationcov::Vector{ExplanatoryVariable} = Vector{ExplanatoryVariable}(),
    logscalecov::Vector{ExplanatoryVariable} = Vector{ExplanatoryVariable}(),
    shapecov::Vector{ExplanatoryVariable} = Vector{ExplanatoryVariable}())::BlockMaxima where T<:Distribution

    return BlockMaxima(data, locationcov = locationcov, logscalecov = logscalecov, shapecov = shapecov, dist = T)
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
    getinitialvalue(::Type{GeneralizedExtremeValue},y::Vector{<:Real})::Vector{<:Real}

Compute the initial values of the GEV parameters given the data `y`.

"""
function getinitialvalue(::Type{GeneralizedExtremeValue},y::Vector{<:Real})::Vector{<:Real}

    # Fit the model with by the probability weigthed moments
    fm = gevfitpwm(y)

    # Convert to fitted model in a Distribution object
    fd = getdistribution(fm.model, fm.θ̂)[]

    # check if initial values are in the domain of the GEV
    valid_initialvalues = all(insupport(fd,y))

    #= If one at least one value does not lie in the support, then the initial
     values are replaced by the Gumbel initial values. =#
    if valid_initialvalues
        μ₀ = location(fd)
        σ₀ = scale(fd)
        ξ₀ = Distributions.shape(fd)
    else
        fm = gumbelfitpwm(y)
        μ₀ = fm.θ̂[1]
        σ₀ = fm.θ̂[2]
        ξ₀ = 0.0
    end

    initialvalues = [μ₀, σ₀, ξ₀]

    return initialvalues

end

"""
    getinitialvalue(model::BlockMaxima)::Vector{<:Real}

Get an initial values vector for the parameters of model.

"""
function getinitialvalue(model::BlockMaxima)::Vector{<:Real}

    y = model.data

    # Compute stationary initial values
    μ₀,σ₀,ξ₀ = getinitialvalue(GeneralizedExtremeValue,y)
    # Store them in a dictionary
    θ₀ = Dict(:μ => μ₀, :ϕ => log(σ₀), :ξ => ξ₀)

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
    println(io, prefix, "data :\t\t", typeof(obj.data), "[", length(obj.data), "]")
    println(io, prefix, "location :\t", showparamfun("μ", obj.location))
    println(io, prefix, "logscale :\t", showparamfun("ϕ", obj.logscale))
    println(io, prefix, "shape :\t\t", showparamfun("ξ", obj.shape))

end
