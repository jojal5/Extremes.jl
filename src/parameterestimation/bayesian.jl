"""
    fitbayes(model::EVA; niter::Int=5000, warmup::Int=2000)::BayesianEVA

Fit the extreme value model under the Bayesian paradigm.

"""
function fitbayes(model::EVA; niter::Int=5000, warmup::Int=2000)::BayesianEVA

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
        sample!(θ, adapt = (i <= warmup))
        if i > warmup
            sim[i, :, 1] = θ
        end
    end

    fittedmodel = BayesianEVA(model, sim)

    return fittedmodel

end

include(joinpath("bayesian", "bayesian_gev.jl"))
include(joinpath("bayesian", "bayesian_gp.jl"))
