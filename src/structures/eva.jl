abstract type EVA end

struct paramfun
    covariate::Vector{<:DataItem}
    fun::Function
end

Base.Broadcast.broadcastable(obj::EVA) = Ref(obj)

"""
    computeparamfunction(covariates::Vector{Variable})::Function

Establish the parameter as function of the corresponding covariates.

"""
function computeparamfunction(covariates::Vector{<:DataItem})::Function

    fun =
    if isempty(covariates)
        function(β::Vector{<:Real})
            return identity(β)
        end
    else
        X = ones(length(covariates[1].value))

        for cov in covariates
            X = hcat(X, cov.value)
        end
        function(β::Vector{<:Real})
            return X*β
        end
    end
    return fun

end

"""
    loglike(model::EVA, θ::Vector{<:Real})::Real

Compute the model loglikelihood evaluated at θ.

"""
function loglike(model::EVA, θ::Vector{<:Real})::Real

    y = model.data.value

    pd = getdistribution(model, θ)

    ll = sum(logpdf.(pd, y))

    return ll

end

"""
    quantile(model::EVA, θ::Vector{<:Real}, p::Real):Vector{<:Real}

Compute the quantile of level `p` from the model evaluated at `θ"". If the model is non-stationary, then the effective quantiles are returned.

"""
function quantile(model::EVA, θ::Vector{<:Real}, p::Real)::Vector{<:Real}

    @assert zero(p)<p<one(p) "the quantile level should be between 0 and 1."

    pd = getdistribution(model, θ)

    q = quantile.(pd, p)

    return q

end

"""
    validatestationarity(model::T)::T where T<:EVA

Warns that the non-stationarity won't be taken into account for this fit and returns a stationary model.

"""
function validatestationarity(model::T)::T where T<:EVA

    if getcovariatenumber(model) > 0
        @warn "covariates cannot be included in the model when estimating the
            paramters by the probability weighted moment parameter estimation.
            The estimates for the stationary model is returned."

        return T(model.data)
    end

    return model

end

"""
    Base.show(io::IO, obj::EVA)

Override of the show function for the objects of type EVA.

"""
function Base.show(io::IO, obj::EVA)

    showEVA(io, obj)

end

"""
    showparamfun(name::String, param::paramfun)::String

Constructs a string describing a parameter `param` with name `name`.

"""
function showparamfun(name::String, param::paramfun)::String

    covariate = [" + $(x.name)" for x in param.covariate]
    res = string("$name ~ 1", covariate...)

    return res

end

"""
    validatelength(length::Real, explvariables::Vector{Variable})

Validates that the values of the explanatory variables are of length `length`.
"""
function validatelength(n::Real, explvariables::Vector{<:DataItem})

    for explvariable in explvariables
        @assert length(explvariable.value) == n "The explanatory variable length should match data length."
    end

end


include(joinpath("eva", "blockmaxima.jl"))
include(joinpath("eva", "thresholdexceedance.jl"))
