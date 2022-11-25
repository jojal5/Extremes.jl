"""
    generalizedloglike(model::EVA, θ::Vector{<:Real})
Compute the model generalized loglikelihood evaluated at θ.
"""
function generalizedloglike(model::EVA, θ::Vector{<:Real}, shapeprior::Distribution)
       
    y = model.data.value
    
    @assert length(θ) == Extremes.nparameter(model) "The length of the parameter vector should be equal to the model number of parameters."

    pi = Extremes.paramindex(model)
    μ = model.location.fun(θ[pi[:μ]])
    ϕ = model.logscale.fun(θ[pi[:ϕ]])
    ξ = model.shape.fun(θ[pi[:ξ]])

    @assert size(unique(ξ), 1) == 1 "Generalized Maximum Likelihood estimation only supports stationnary ξ."
    σ = exp.(ϕ)

    pd = GeneralizedExtremeValue.(μ, σ, ξ)

    ll₁ = sum(logpdf.(pd, y))
    ll₂ = logpdf(shapeprior, ξ[1])
    
    return ll₁ + ll₂
    
end

"""
    gmlefit(model::EVA; initialvalues::Vector{<:Real}; α::Float64=6.0, β::Float64=9.0))::GeneralizedMaximumLikelihoodEVA
Fit the extreme value model by Generalized maximum likelihood.
"""
function fitgmle(model::EVA, initialvalues::Vector{<:Real}; shapeprior::Distribution=LocationScale(-.5, 1, Beta(6, 9)))::GeneralizedMaximumLikelihoodEVA
    
    # Block maxima model validation
    @assert typeof(model) == BlockMaxima "Model not supported."

    # Initial values validation
    fd = Extremes.getdistribution(model, initialvalues)
    @assert all(insupport.(fd, model.data.value)) "The initial value vector is not a member of the set of possible solutions. At least one data lies outside the distribution support."

    fobj(θ) = -generalizedloglike(model, θ, shapeprior)
    res = Optim.optimize(fobj, initialvalues)

    if Optim.converged(res)
        θ̂ = Optim.minimizer(res)
    else
        @warn "The generalized maximum likelihood algorithm did not find a solution. Maybe try with different initial values or with another method. The returned values are the initial values."
        θ̂ = initialvalues
    end

    fittedmodel = GeneralizedMaximumLikelihoodEVA(model, θ̂)

    return fittedmodel

end

"""
    fitgmle(model::EVA)::MaximumLikelihoodEVA
Fit the extreme value model by maximum likelihood.
"""
function fitgmle(model::EVA; shapeprior::Distribution=LocationScale(-.5, 1, Beta(6, 9)))::GeneralizedMaximumLikelihoodEVA

    initialvalues = getinitialvalue(model)

    return fitgmle(model, initialvalues, shapeprior)

end

include(joinpath("generalizedmaximumlikelihood", "generalizedmaximumlikelihood_gev.jl"))