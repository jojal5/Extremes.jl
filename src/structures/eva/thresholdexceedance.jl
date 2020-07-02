struct ThresholdExceedance <: EVA
    data::Variable
    logscale::paramfun
    shape::paramfun
end

"""
    ThresholdExceedance(exceedances::Vector{<:Real};
        logscalecov::Vector{<:DataItem} = Vector{Variable}(),
        shapecov::Vector{<:DataItem} = Vector{Variable}())::ThresholdExceedance

Creates a ThresholdExceedance structure.

"""
function ThresholdExceedance(exceedances::Variable;
    logscalecov::Vector{<:DataItem} = Vector{Variable}(),
    shapecov::Vector{<:DataItem} = Vector{Variable}())::ThresholdExceedance

    n = length(exceedances.value)
    validatelength(n, logscalecov)
    validatelength(n, shapecov)

    logscalefun = computeparamfunction(logscalecov)
    shapefun = computeparamfunction(shapecov)

    return ThresholdExceedance(exceedances, paramfun(logscalecov, logscalefun), paramfun(shapecov, shapefun))

end

"""
    paramindex(model::ThresholdExceedance)::Dict{Symbol,Vector{<:Int}}

Get the parameter indexing for a ThresholdExceedance.

"""
function paramindex(model::ThresholdExceedance)::Dict{Symbol,Vector{<:Int}}

    i = 0
    function increasei()
        i = i + 1
        return i
    end
    return Dict{Symbol,Vector{<:Int}}(
        :ϕ => Int64[increasei() for k in 1:length(model.logscale.covariate) + 1],
        :ξ => Int64[increasei() for k in 1:length(model.shape.covariate) + 1]
    )

end

"""
    getcovariatenumber(model::ThresholdExceedance)::Int

Return the number of covariates.

"""
function getcovariatenumber(model::ThresholdExceedance)::Int

    return sum([length(model.logscale.covariate), length(model.shape.covariate)])

end

"""
    nparameter(model::ThresholdExceedance)::Int

Get the number of parameters in a ThresholdExceedance.

"""
function nparameter(model::ThresholdExceedance)::Int

    return 2 + getcovariatenumber(model)

end

"""
    getdistribution(model::ThresholdExceedance, θ::Vector{<:Real})::Vector{<:Distribution}

Return the fitted distribution in case of stationarity or the vector of fitted distribution in case of non-stationarity.

"""
function getdistribution(model::ThresholdExceedance, θ::AbstractVector{<:Real})::Vector{<:Distribution}

    @assert length(θ)==nparameter(model) "The length of the parameter vector should be equal to the model number of parameters."

    pi = paramindex(model)
    ϕ = model.logscale.fun(θ[pi[:ϕ]])
    ξ = model.shape.fun(θ[pi[:ξ]])

    σ = exp.(ϕ)

    fd = GeneralizedPareto.(σ, ξ)

    return fd

end


"""
    getinitialvalue(model::ThresholdExceedance)::Vector{<:Real}

Get an initial values vector for the parameters of model.

"""
function getinitialvalue(model::ThresholdExceedance)::Vector{<:Real}

    y = model.data.value

    # Compute stationary initial values
    ϕ₀,ξ₀ = getinitialvalue(GeneralizedPareto, y)
    # Store them in a dictionary
    θ₀ = Dict(:ϕ => ϕ₀, :ξ => ξ₀)

    initialvalues = zeros(nparameter(model))
    pi = paramindex(model)
    for param in [:ϕ, :ξ]
        ind = pi[param][1]
        initialvalues[ind] = θ₀[param]
    end

    return initialvalues

end

"""
    showEVA(io::IO, obj::ThresholdExceedance; prefix::String = "")

Displays a ThresholdExceedance with the prefix `prefix` before every line.

"""
function showEVA(io::IO, obj::ThresholdExceedance; prefix::String = "")

    println(io, prefix, "ThresholdExceedance")
    println(io, prefix, "data :\t\t",typeof(obj.data.value), "[", length(obj.data.value), "]")
    println(io, prefix, "logscale :\t", showparamfun("ϕ", obj.logscale))
    println(io, prefix, "shape :\t\t", showparamfun("ξ", obj.shape))

end
