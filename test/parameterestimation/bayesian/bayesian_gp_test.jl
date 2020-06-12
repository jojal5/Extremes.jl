@testset "bayesian_gp.jl" begin
    n = 10000

    x₁ = randn(n) / 3
    x₂ = randn(n) / 10

    ϕ = -.05 .+ x₁
    ξ = x₂

    σ = exp.(ϕ)
    θ = [-0.05; 1.0; 0.0; 1.0]

    pd = GeneralizedPareto.(σ, ξ)
    y = rand.(pd)

    @testset "gpfitbayes(y; logscalecov, shapecov, niter, warmup)" begin
        # model building with non-stationary logscale and shape
        fm = Extremes.gpfitbayes(y,
            logscalecov = [ExplanatoryVariable("x₁", x₁)],
            shapecov = [ExplanatoryVariable("x₂", x₂)],
            niter=1000, warmup=500)

        θ̂ = dropdims(mean(fm.sim.value[:,:,1], dims=1)',dims=2)

        @test θ̂ ≈ θ atol = .1
    end

    @testset "gpfitbayes(df, datacol; logscalecovid, shapecovid, niter, warmup)" begin
        # model building with non-stationary location, logscale and shape
        df = DataFrame(y = y, x1 = x₁, x2 = x₂)

        fm = Extremes.gpfitbayes(df, :y, logscalecovid = [:x1], shapecovid = [:x2], niter=1000, warmup=500)

        θ̂ = dropdims(mean(fm.sim.value[:,:,1], dims=1)',dims=2)

        @test θ̂ ≈ θ atol = .1

    end

    @testset "gpfitbayes(model; niter, warmup)" begin
        # non-stationary location, logscale and shape
        model = Extremes.ThresholdExceedance(y,
            logscalecov = [ExplanatoryVariable("x₁", x₁)],
            shapecov = [ExplanatoryVariable("x₂", x₂)])

        fm = Extremes.gpfitbayes(model, niter=1000, warmup=500)

        θ̂ = dropdims(mean(fm.sim.value[:,:,1], dims=1)',dims=2)

        @test θ̂ ≈ θ atol = .1

    end

end
