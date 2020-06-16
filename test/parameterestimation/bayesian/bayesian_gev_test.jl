@testset "bayesian_gev.jl" begin
    n = 10000

    x₁ = randn(n)
    x₂ = randn(n) / 3
    x₃ = randn(n) / 10

    μ = 1.0 .+ x₁
    ϕ = -.05 .+ x₂
    ξ = x₃

    σ = exp.(ϕ)
    θ = [1.0; 1.0; -0.05; 1.0; 0.0; 1.0]

    pd = GeneralizedExtremeValue.(μ, σ, ξ)
    y = rand.(pd)

    @testset "gevfitbayes(y; locationcov, logscalecov, shapecov, niter, warmup)" begin
        # model building with non-stationary location, logscale and shape
        fm = Extremes.gevfitbayes(y,
            locationcov = [ExplanatoryVariable("x₁", x₁)],
            logscalecov = [ExplanatoryVariable("x₂", x₂)],
            shapecov = [ExplanatoryVariable("x₃", x₃)],
            niter=1000, warmup=500)

        infq = quantile!.([fm.sim.value[:,:,1][:,i] for i in 1:length(θ)], 0.025)
        supq = quantile!.([fm.sim.value[:,:,1][:,i] for i in 1:length(θ)], 0.975)

        @test infq <= θ
        @test θ <= supq

    end

    @testset "gevfitbayes(df, datacol; locationcovid, logscalecovid, shapecovid, niter, warmup)" begin
        # model building with non-stationary location, logscale and shape
        df = DataFrame(y = y, x1 = x₁, x2 = x₂, x3 = x₃)

        fm = Extremes.gevfitbayes(df, :y, locationcovid = [:x1], logscalecovid = [:x2], shapecovid = [:x3], niter=1000, warmup=500)

        infq = quantile!.([fm.sim.value[:,:,1][:,i] for i in 1:length(θ)], 0.025)
        supq = quantile!.([fm.sim.value[:,:,1][:,i] for i in 1:length(θ)], 0.975)

        @test infq <= θ
        @test θ <= supq

    end

    @testset "gevfitbayes(model; niter, warmup)" begin
        # non-stationary location, logscale and shape
        model = Extremes.BlockMaxima(y,
            locationcov = [ExplanatoryVariable("x₁", x₁)],
            logscalecov = [ExplanatoryVariable("x₂", x₂)],
            shapecov = [ExplanatoryVariable("x₃", x₃)])

        fm = Extremes.gevfitbayes(model, niter=1000, warmup=500)

        infq = quantile!.([fm.sim.value[:,:,1][:,i] for i in 1:length(θ)], 0.025)
        supq = quantile!.([fm.sim.value[:,:,1][:,i] for i in 1:length(θ)], 0.975)

        @test infq <= θ
        @test θ <= supq

    end

end
