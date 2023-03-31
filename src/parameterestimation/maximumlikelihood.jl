"""
    fit(model::AbstractExtremeValueModel; initialvalues::Vector{<:Real})::MaximumLikelihoodAbstractExtremeValueModel

Fit the extreme value model by maximum likelihood.

"""
function fit(model::AbstractExtremeValueModel, initialvalues::Vector{<:Real})::MaximumLikelihoodAbstractExtremeValueModel

    # Initial values validation
    fd = getdistribution(model, initialvalues)
    @assert all(insupport.(fd, model.data.value)) "The initial value vector is not a member of the set of possible solutions. At least one data lies outside the distribution support."

    fobj(θ) = -loglike(model, θ)

    res = optimize(fobj, initialvalues)

    if Optim.converged(res)
        θ̂ = Optim.minimizer(res)
    else
        @warn "The maximum likelihood algorithm did not find a solution. Maybe try with different initial values or with another method. The returned values are the initial values."
        θ̂ = initialvalues
    end

    fittedmodel = MaximumLikelihoodAbstractExtremeValueModel(model, θ̂)

    return fittedmodel

end

"""
    fit(model::AbstractExtremeValueModel)::MaximumLikelihoodAbstractExtremeValueModel

Fit the extreme value model by maximum likelihood.

"""
function fit(model::AbstractExtremeValueModel)::MaximumLikelihoodAbstractExtremeValueModel

    initialvalues = getinitialvalue(model)

    return fit(model, initialvalues)

end

include(joinpath("maximumlikelihood", "maximumlikelihood_gev.jl"))
include(joinpath("maximumlikelihood", "maximumlikelihood_gp.jl"))
include(joinpath("maximumlikelihood", "maximumlikelihood_gumbel.jl"))
