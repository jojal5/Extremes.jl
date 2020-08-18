@testset "bayesian_gp.jl" begin
    n = 100

    x₁ = Variable("x₁", randn(n) / 3)
    x₂ = Variable("x₂", randn(n) / 10)

    ϕ = -.05 .+ x₁.value
    ξ = x₂.value

    σ = exp.(ϕ)
    θ = [-0.05; 1.0; 0.0; 1.0]

    pd = GeneralizedPareto.(σ, ξ)
    y = rand.(pd)

    @testset "gpfitbayes(y; logscalecov, shapecov, niter, warmup)" begin
        # model building with non-stationary logscale and shape
        fm = Extremes.gpfitbayes(y,
            logscalecov = [x₁],
            shapecov = [x₂],
            niter=500, warmup=5)

            # data is y
            @test fm.model.data.value ≈ y

            # logscale is x₁
            @test length(fm.model.logscale.covariate) == 1
            @test fm.model.logscale.covariate[1].value ≈ x₁.value

            # shape is x₂
            @test length(fm.model.shape.covariate) == 1
            @test fm.model.shape.covariate[1].value ≈ x₂.value

    end

    @testset "gpfitbayes(df, datacol; logscalecovid, shapecovid, niter, warmup)" begin
        # model building with non-stationary location, logscale and shape
        df = DataFrame(y = y, x1 = x₁.value, x2 = x₂.value)

        fm = Extremes.gpfitbayes(df, :y, logscalecovid = [:x1], shapecovid = [:x2], niter=500, warmup=5)

        # data is y
        @test fm.model.data.value ≈ y

        # logscale is x₁
        @test length(fm.model.logscale.covariate) == 1
        @test fm.model.logscale.covariate[1].value ≈ x₁.value

        # shape is x₂
        @test length(fm.model.shape.covariate) == 1
        @test fm.model.shape.covariate[1].value ≈ x₂.value

    end

    @testset "gpfitbayes(model; niter, warmup)" begin
        # non-stationary location, logscale and shape
        model = Extremes.ThresholdExceedance(Variable("y", y),
            logscalecov = [x₁],
            shapecov = [x₂])

        fm = Extremes.gpfitbayes(model, niter=500, warmup=5)

        # data is y
        @test fm.model.data.value ≈ y

        # logscale is x₁
        @test length(fm.model.logscale.covariate) == 1
        @test fm.model.logscale.covariate[1].value ≈ x₁.value

        # shape is x₂
        @test length(fm.model.shape.covariate) == 1
        @test fm.model.shape.covariate[1].value ≈ x₂.value

    end

end
