@testset "bayesian_gev.jl" begin
    n = 100

    x₁ = Variable("x₁", randn(n))
    x₂ = Variable("x₂", randn(n) / 3)
    x₃ = Variable("x₃", randn(n) / 10)

    μ = 1.0 .+ x₁.value
    ϕ = -.05 .+ x₂.value
    ξ = x₃.value

    σ = exp.(ϕ)
    θ = [1.0; 1.0; -0.05; 1.0; 0.0; 1.0]

    pd = GeneralizedExtremeValue.(μ, σ, ξ)
    y = rand.(pd)

    @testset "gevfitbayes(y; locationcov, logscalecov, shapecov, niter, warmup)" begin
        # model building with non-stationary location, logscale and shape
        fm = Extremes.gevfitbayes(y,
            locationcov = [x₁],
            logscalecov = [x₂],
            shapecov = [x₃],
            niter=500, warmup=5)

        # data is y
        @test fm.model.data.value ≈ y

        # location is x₁
        @test length(fm.model.location.covariate) == 1
        @test fm.model.location.covariate[1].value ≈ x₁.value

        # logscale is x₂
        @test length(fm.model.logscale.covariate) == 1
        @test fm.model.logscale.covariate[1].value ≈ x₂.value

        # shape is x₃
        @test length(fm.model.shape.covariate) == 1
        @test fm.model.shape.covariate[1].value ≈ x₃.value

    end

    @testset "gevfitbayes(df, datacol; locationcovid, logscalecovid, shapecovid, niter, warmup)" begin
        # model building with non-stationary location, logscale and shape
        df = DataFrame(y = y, x1 = x₁.value, x2 = x₂.value, x3 = x₃.value)

        fm = Extremes.gevfitbayes(df, :y, locationcovid = [:x1], logscalecovid = [:x2], shapecovid = [:x3], niter=500, warmup=5)

        # data is y
        @test fm.model.data.value ≈ y

        # location is x₁
        @test length(fm.model.location.covariate) == 1
        @test fm.model.location.covariate[1].value ≈ x₁.value

        # logscale is x₂
        @test length(fm.model.logscale.covariate) == 1
        @test fm.model.logscale.covariate[1].value ≈ x₂.value

        # shape is x₃
        @test length(fm.model.shape.covariate) == 1
        @test fm.model.shape.covariate[1].value ≈ x₃.value

    end

    @testset "gevfitbayes(model; niter, warmup)" begin
        # non-stationary location, logscale and shape
        model = Extremes.BlockMaxima{GeneralizedExtremeValue}(Variable("y", y),
            locationcov = [x₁],
            logscalecov = [x₂],
            shapecov = [x₃])

        fm = Extremes.gevfitbayes(model, niter=500, warmup=5)

        # data is y
        @test fm.model.data.value ≈ y

        # location is x₁
        @test length(fm.model.location.covariate) == 1
        @test fm.model.location.covariate[1].value ≈ x₁.value

        # logscale is x₂
        @test length(fm.model.logscale.covariate) == 1
        @test fm.model.logscale.covariate[1].value ≈ x₂.value

        # shape is x₃
        @test length(fm.model.shape.covariate) == 1
        @test fm.model.shape.covariate[1].value ≈ x₃.value

    end

end
