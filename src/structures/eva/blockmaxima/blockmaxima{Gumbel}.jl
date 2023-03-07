
"""
    BlockMaxima{Gumbel}(data::Vector{<:Real};
        locationcov::Vector{Variable} = Vector{Variable}(),
        logscalecov::Vector{Variable} = Vector{Variable}())::BlockMaxima

Creates a BlockMaxima{Gumbel} structure.

"""
function BlockMaxima{Gumbel}(data::Variable;
    locationcov::Vector{<:DataItem} = Vector{Variable}(),
    logscalecov::Vector{<:DataItem} = Vector{Variable}())::BlockMaxima

    # No shape non-stationarity for the Gumbel model
    shapecov = Vector{Variable}()

    n = length(data.value)
    validatelength(n, locationcov)
    validatelength(n, logscalecov)

    locationfun = computeparamfunction(locationcov)
    logscalefun = computeparamfunction(logscalecov)
    shapefun = computeparamfunction(shapecov)

    return BlockMaxima{Gumbel}(data, paramfun(locationcov, locationfun), paramfun(logscalecov, logscalefun), paramfun(shapecov, shapefun))

end


function getcovariatenumber(model::BlockMaxima{Gumbel})::Int

    return sum([length(model.location.covariate), length(model.logscale.covariate)])

end


function getdistribution(model::BlockMaxima{Gumbel}, θ::AbstractVector{<:Real})::Vector{<:Distribution}

    @assert length(θ)==nparameter(model) "The length of the parameter vector should be equal to the model number of parameters."

    pi = paramindex(model)
    μ = model.location.fun(θ[pi[:μ]])
    ϕ = model.logscale.fun(θ[pi[:ϕ]])

    σ = exp.(ϕ)

    fd = Gumbel.(μ, σ)

    return fd

end

function getinitialvalue(model::BlockMaxima{Gumbel})::Vector{<:Real}

    y = model.data.value

    # Compute stationary initial values
    fm = gumbelfitpwm(y)
    μ₀ = fm.θ̂[1]
    ϕ₀ = fm.θ̂[2]
    # Store them in a dictionary
    θ₀ = Dict(:μ => μ₀, :ϕ => ϕ₀)

    initialvalues = zeros(nparameter(model))
    pi = paramindex(model)
    for param in [:μ, :ϕ]
        ind = pi[param][1]
        initialvalues[ind] = θ₀[param]
    end

    return initialvalues

end


function nparameter(model::BlockMaxima{Gumbel})::Int

    return 2 + getcovariatenumber(model)

end

function paramindex(model::BlockMaxima{Gumbel})::Dict{Symbol,Vector{<:Int}}

    sμ = length(model.location.covariate) + 1
    sϕ = length(model.logscale.covariate) + 1

    return Dict{Symbol,Vector{<:Int}}(
        :μ => Int64[k for k in 1:sμ],
        :ϕ => Int64[k for k in sμ .+ (1:sϕ)]
    )

end


"""
    showEVA(io::IO, obj::BlockMaxima{Gumbel}; prefix::String = "")

Displays a BlockMaxima{Gumbel} with the prefix `prefix` before every line.

"""
function showEVA(io::IO, obj::BlockMaxima{Gumbel}; prefix::String = "")

    println(io, prefix, "BlockMaxima{Gumbel}")
    println(io, prefix, "data :\t\t", typeof(obj.data.value), "[", length(obj.data.value), "]")
    println(io, prefix, "location :\t", showparamfun("μ", obj.location))
    println(io, prefix, "logscale :\t", showparamfun("ϕ", obj.logscale))

end


