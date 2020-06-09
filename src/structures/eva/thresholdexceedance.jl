struct ThresholdExceedance <: EVA
    data::Vector{<:Real}
    logscale::paramfun
    shape::paramfun
end

"""
    ThresholdExceedance(exceedances::Vector{<:Real};
        logscalecov::Vector{ExplanatoryVariable} = Vector{ExplanatoryVariable}(),
        shapecov::Vector{ExplanatoryVariable} = Vector{ExplanatoryVariable}())::ThresholdExceedance

Creates a ThresholdExceedance structure.

"""
function ThresholdExceedance(exceedances::Vector{<:Real};
    logscalecov::Vector{ExplanatoryVariable} = Vector{ExplanatoryVariable}(),
    shapecov::Vector{ExplanatoryVariable} = Vector{ExplanatoryVariable}())::ThresholdExceedance

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
function getdistribution(model::ThresholdExceedance, θ::Vector{<:Real})::Vector{<:Distribution}

    @assert length(θ)==nparameter(model) "The length of the parameter vector should be equal to the model number of parameters."

    pi = paramindex(model)
    ϕ = model.logscale.fun(θ[pi[:ϕ]])
    ξ = model.shape.fun(θ[pi[:ξ]])

    σ = exp.(ϕ)

    fd = GeneralizedPareto.(σ, ξ)

    return fd

end

"""
     getinitialvalue(::Type{GeneralizedPareto},y::Vector{<:Real})::Vector{<:Real}

Compute the initial values of the GP parameters given the data `y`.

"""
function getinitialvalue(::Type{GeneralizedPareto},y::Vector{<:Real})::Vector{<:Real}

    # Fit the model with by the probability weigthed moments
    fm = gpfitpwm(y)

    # Convert to fitted model in a Distribution object
    fd = getdistribution(fm.model, fm.θ̂)[]

    if all(insupport(fd,y))
        σ₀ = scale(fd)
        ξ₀ = Distributions.shape(fd)
    else
        σ₀ = mean(y)
        ξ₀ = 0.0
    end

    initialvalues = [σ₀, ξ₀]

    return initialvalues

end

"""
    getinitialvalue(model::ThresholdExceedance)::Vector{<:Real}

Get an initial values vector for the parameters of model.

"""
function getinitialvalue(model::ThresholdExceedance)::Vector{<:Real}

    y = model.data

    # Compute stationary initial values
    σ₀,ξ₀ = getinitialvalue(GeneralizedPareto, y)
    # Store them in a dictionary
    θ₀ = Dict(:ϕ => log(σ₀), :ξ => ξ₀)

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
    println(io, prefix, "data :\t\t",typeof(obj.data), "[", length(obj.data), "]")
    println(io, prefix, "logscale :\t", showparamfun("ϕ", obj.logscale))
    println(io, prefix, "shape :\t\t", showparamfun("ξ", obj.shape))

end
