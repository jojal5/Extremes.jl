@testset "maximumlikelihood_gp.jl" begin
    n = 10000

    x₁ = randn(n) / 3
    x₂ = randn(n) / 10

    ϕ = -.05 .+ x₁
    ξ = x₂

    σ = exp.(ϕ)
    θ = [-0.05; 1.0; 0.0; 1.0]

    pd = GeneralizedPareto.(σ, ξ)
    y = rand.(pd)

    @testset "gpfit(y; logscalecov, shapecov)" begin
        # model building with non-stationary logscale and shape
        fm = Extremes.gpfit(y,
            logscalecov = [ExplanatoryVariable("x₁", x₁)],
            shapecov = [ExplanatoryVariable("x₂", x₂)])

            @test fm.θ̂ ≈ θ atol = 0.1

    end

    @testset "gpfit(df, datacol; logscalecovid, shapecovid)" begin
        # model building with non-stationary location, logscale and shape
        df = DataFrame(y = y, x1 = x₁, x2 = x₂)

        fm = Extremes.gpfit(df, :y, logscalecovid = [:x1], shapecovid = [:x2])

        @test fm.θ̂ ≈ θ atol = 0.1

    end

    @testset "gpfit(model)" begin
        # non-stationary location, logscale and shape
        model = Extremes.ThresholdExceedance(y,
            logscalecov = [ExplanatoryVariable("x₁", x₁)],
            shapecov = [ExplanatoryVariable("x₂", x₂)])

        fm = Extremes.gpfit(model)

        @test fm.θ̂ ≈ θ atol = 0.1

    end

end
