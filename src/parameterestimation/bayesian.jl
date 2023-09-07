"""
    fitbayes(model::AbstractExtremeValueModel; niter::Int=5000, warmup::Int=2000)::BayesianAbstractExtremeValueModel

Fit the extreme value model under the Bayesian paradigm.

"""
function fitbayes(model::AbstractExtremeValueModel; niter::Int=5000, warmup::Int=2000)::BayesianAbstractExtremeValueModel

    # Set initial values to the maximum likelihood estimates
    ml = fit(model)
    initialvalues = ml.θ̂

    # Define the loglikelihood function and the gradient for the NUTS algorithm
    logf(θ::DenseVector) = loglike(model,θ)
    Δlogf(θ::DenseVector) = ForwardDiff.gradient(logf, θ)
    function logfgrad(θ::DenseVector)
        ll = logf(θ)
        g = Δlogf(θ)
        return ll, g
    end

    sim = Chains(niter, nparameter(model), start = (warmup + 1))
    θ = NUTSVariate(initialvalues, logfgrad)
    @showprogress for i in 1:niter
        MambaLite.sample!(θ, adapt = (i <= warmup))
        if i > warmup
            sim[i, :, 1] = θ
        end
    end

    fittedmodel = BayesianAbstractExtremeValueModel(model, sim)

    return fittedmodel

end

include(joinpath("bayesian", "bayesian_gev.jl"))
include(joinpath("bayesian", "bayesian_gp.jl"))
include(joinpath("bayesian", "bayesian_gumbel.jl"))
