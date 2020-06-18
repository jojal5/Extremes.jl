@testset "bayesian.jl" begin
    @testset "fitbayes(model; niter, warmup)" begin

        # stationary GEV Bayesian fit
        n = 5000

        μ = 0.0
        σ = 1.0
        ξ = 0.1

        ϕ = log(σ)
        θ = [μ; ϕ; ξ]

        pd = GeneralizedExtremeValue(μ, σ, ξ)
        y = rand(pd, n)

        model = Extremes.BlockMaxima(y)

        fm = Extremes.fitbayes(model, niter=2000, warmup=1000)

        infq = quantile!.([fm.sim.value[:,:,1][:,i] for i in 1:length(θ)], 0.025)
        supq = quantile!.([fm.sim.value[:,:,1][:,i] for i in 1:length(θ)], 0.975)

        @test infq <= θ
        @test θ <= supq


        # non-stationary location GEV Bayesian fit
        n = 10000

        x₁ = randn(n)
        x₂ = randn(n)

        μ = x₁ + x₂
        σ = 1.0
        ξ = 0.1

        ϕ = log(σ)
        θ = [0.0; 1.0; 1.0; ϕ; ξ]

        pd = GeneralizedExtremeValue.(μ, σ, ξ)
        y = rand.(pd)

        model = Extremes.BlockMaxima(y, locationcov = [Variable("x₁", x₁), Variable("x₂", x₂)])

        fm = Extremes.fitbayes(model, niter=2000, warmup=1000)

        infq = quantile!.([fm.sim.value[:,:,1][:,i] for i in 1:length(θ)], 0.025)
        supq = quantile!.([fm.sim.value[:,:,1][:,i] for i in 1:length(θ)], 0.975)

        @test infq <= θ
        @test θ <= supq

        # non-stationary logscale GEV Bayesian fit
        n = 10000

        x₁ = randn(n) / 3
        x₂ = randn(n) / 3

        μ = 1.0
        ϕ = x₁ + x₂
        ξ = 0.1

        σ = exp.(ϕ)
        θ = [μ; 0.0; 1.0; 1.0; ξ]

        pd = GeneralizedExtremeValue.(μ, σ, ξ)
        y = rand.(pd)

        model = Extremes.BlockMaxima(y, logscalecov = [Variable("x₁", x₁), Variable("x₂", x₂)])

        fm = Extremes.fitbayes(model, niter=2000, warmup=1000)

        infq = quantile!.([fm.sim.value[:,:,1][:,i] for i in 1:length(θ)], 0.025)
        supq = quantile!.([fm.sim.value[:,:,1][:,i] for i in 1:length(θ)], 0.975)

        @test infq <= θ
        @test θ <= supq

        # non-stationary location and logscale GEV Bayesian fit
        n = 10000

        x₁ = randn(n)
        x₂ = randn(n) / 3

        μ = 1.0 .+ x₁
        ϕ = -.05 .+ x₂
        ξ = 0.1

        σ = exp.(ϕ)
        θ = [1.0; 1.0; -.05; 1.0; ξ]

        pd = GeneralizedExtremeValue.(μ, σ, ξ)
        y = rand.(pd)

        model = Extremes.BlockMaxima(y, locationcov = [Variable("x₁", x₁)], logscalecov = [Variable("x₂", x₂)])

        fm = Extremes.fitbayes(model, niter=2000, warmup=1000)

        infq = quantile!.([fm.sim.value[:,:,1][:,i] for i in 1:length(θ)], 0.025)
        supq = quantile!.([fm.sim.value[:,:,1][:,i] for i in 1:length(θ)], 0.975)

        @test infq <= θ
        @test θ <= supq

        # non-stationary shape GEV Bayesian fit
        # n = 100000
        #
        # x₁ = randn(n) / 10
        #
        # μ = 0.0
        # ϕ = 0.0
        # ξ = x₁
        #
        # σ = exp(ϕ)
        # θ = [μ; ϕ; 0.0; 1.0]
        #
        # pd = GeneralizedExtremeValue.(μ, σ, ξ)
        # y = rand.(pd)
        #
        # model = Extremes.BlockMaxima(y, shapecov = [Variable("x₁", x₁)])
        #
        # fm = Extremes.fitbayes(model, niter=2000, warmup=1000)
        #
        #infq = quantile!.([fm.sim.value[:,:,1][:,i] for i in 1:length(θ)], 0.025)
        #supq = quantile!.([fm.sim.value[:,:,1][:,i] for i in 1:length(θ)], 0.975)

        #@test infq <= θ
        #@test θ <= supq5

        # stationary GP bayes fit
        n = 10000

        σ = 1.0
        ξ = 0.1

        ϕ = log(σ)
        θ = [ϕ; ξ]

        pd = GeneralizedPareto(σ, ξ)
        y = rand(pd, n)

        model = Extremes.ThresholdExceedance(y)

        fm = Extremes.fitbayes(model, niter=2000, warmup=1000)

        infq = quantile!.([fm.sim.value[:,:,1][:,i] for i in 1:length(θ)], 0.025)
        supq = quantile!.([fm.sim.value[:,:,1][:,i] for i in 1:length(θ)], 0.975)

        @test infq <= θ
        @test θ <= supq

        # non-stationary logscale GP Bayesian fit
        n = 10000

        x₁ = randn(n) / 3
        x₂ = randn(n) / 3

        ϕ = -.5 .+ x₁ .+ x₂
        ξ = 0.1

        σ = exp.(ϕ)
        θ = [-.5; 1.0; 1.0; ξ]

        pd = GeneralizedPareto.(σ, ξ)
        y = rand.(pd)

        model = Extremes.ThresholdExceedance(y, logscalecov = [Variable("x₁", x₁), Variable("x₂", x₂)])

        fm = Extremes.fitbayes(model, niter=2000, warmup=1000)

        infq = quantile!.([fm.sim.value[:,:,1][:,i] for i in 1:length(θ)], 0.025)
        supq = quantile!.([fm.sim.value[:,:,1][:,i] for i in 1:length(θ)], 0.975)

        @test infq <= θ
        @test θ <= supq

        # # non-stationary shape GP Bayesian fit
        # n = 100000
        #
        # x₁ = randn(n) / 10
        #
        # μ = 0.0
        # ϕ = 0.0
        # ξ = x₁
        #
        # σ = exp(ϕ)
        # θ = [ϕ; 0.0; 1.0]
        #
        # pd = GeneralizedPareto.(μ, σ, ξ)
        # y = rand.(pd)
        #
        # model = Extremes.ThresholdExceedance(y, shapecov = [Variable("x₁", x₁)])
        #
        # fm = Extremes.fitbayes(model, niter=2000, warmup=1000)
        #
        #infq = quantile!.([fm.sim.value[:,:,1][:,i] for i in 1:length(θ)], 0.025)
        #supq = quantile!.([fm.sim.value[:,:,1][:,i] for i in 1:length(θ)], 0.975)

        #@test infq <= θ
        #@test θ <= supq

    end

    include(joinpath("bayesian", "bayesian_gev_test.jl"))
    include(joinpath("bayesian", "bayesian_gp_test.jl"))

end
