"""
    fitbayes(model::EVA; niter::Int=5000, warmup::Int=2000)::BayesianEVA
Fit the extreme value model under the Bayesian paradigm.
"""
function fitbayes(model::EVA; niter::Int=5000, warmup::Int=2000)::BayesianEVA
    
    # Choose parameter dimensionality 
    D = Extremes.nparameter(model)
    
    # Set initial values to the maximum likelihood estimates
    ml = fit(model)
    initialvalues = ml.θ̂
    
    # Define the target distribution
    target(θ) = loglike(model,θ)
    
    # Define a Hamiltonian system
    metric = DiagEuclideanMetric(D)
    hamiltonian = Hamiltonian(metric, target, ForwardDiff)

    # Define a leapfrog solver, with initial step size chosen heuristically
    initial_ϵ = find_good_stepsize(hamiltonian, initialvalues)
    integrator = Leapfrog(initial_ϵ)

    # Define an HMC sampler, with the following components
    #   - Original NUTS with slice sampling
    #   - windowed adaption for step-size and diagonal mass matrix
    proposal = NUTS{SliceTS,ClassicNoUTurn}(integrator)
    adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))

    # Run the sampler to draw samples from the specified model, where
    #   - `samples` will store the samples
    #   - `stats` will store diagnostic statistics for each sample
    samples, stats = sample(hamiltonian, proposal, initialvalues, niter, adaptor, warmup; drop_warmup=true, verbose=false, progress=false);
    
    chn = Array{Float64}(undef, length(samples), length(samples[1]), 1)
    for i in 1:length(samples)
        chn[i, :, 1] = samples[i]
    end
    sim = MCMCChains.Chains(chn, start = (warmup + 1))

    fittedmodel = BayesianEVA(model, sim)

    return fittedmodel

end

include(joinpath("bayesian", "bayesian_gev.jl"))
include(joinpath("bayesian", "bayesian_gp.jl"))
