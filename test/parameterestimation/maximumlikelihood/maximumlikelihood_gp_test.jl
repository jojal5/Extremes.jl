@testset "maximumlikelihood_gp.jl" begin
    n = 100

    x₁ = Variable("x₁", randn(n) / 3)
    x₂ = Variable("x₂", randn(n) / 10)

    ϕ = -.05 .+ x₁.value
    ξ = x₂.value

    σ = exp.(ϕ)
    θ = [-0.05; 1.0; 0.0; 1.0]

    pd = GeneralizedPareto.(σ, ξ)
    y = rand.(pd)

    @testset "gpfit(y; logscalecov, shapecov)" begin
        # model building with non-stationary logscale and shape
        fm = Extremes.gpfit(y,
            logscalecov = [x₁],
            shapecov = [x₂])

        # data is y
        @test fm.model.data.value ≈ y

        # logscale is x₁
        @test length(fm.model.logscale.covariate) == 1
        @test fm.model.logscale.covariate[1].value ≈ x₁.value

        # shape is x₂
        @test length(fm.model.shape.covariate) == 1
        @test fm.model.shape.covariate[1].value ≈ x₂.value

    end

    @testset "gpfit(y, initialvalues; logscalecov, shapecov)" begin
        # model building with non-stationary logscale and shape
        @test_logs Extremes.gpfit(y, zeros(Float64, 4),
            logscalecov = [x₁],
            shapecov = [x₂])

    end

    @testset "gpfit(df, datacol; logscalecovid, shapecovid)" begin
        # model building with non-stationary location, logscale and shape
        df = DataFrame(y = y, x1 = x₁.value, x2 = x₂.value)

        fm = Extremes.gpfit(df, :y, logscalecovid = [:x1], shapecovid = [:x2])

        # data is y
        @test fm.model.data.value ≈ y

        # logscale is x₁
        @test length(fm.model.logscale.covariate) == 1
        @test fm.model.logscale.covariate[1].value ≈ x₁.value

        # shape is x₂
        @test length(fm.model.shape.covariate) == 1
        @test fm.model.shape.covariate[1].value ≈ x₂.value

    end

    @testset "gpfit(df, datacol, initialvalues; logscalecovid, shapecovid)" begin
        # model building with non-stationary location, logscale and shape
        df = DataFrame(y = y, x1 = x₁.value, x2 = x₂.value)

        @test_logs Extremes.gpfit(df, :y, zeros(Float64, 4), logscalecovid = [:x1], shapecovid = [:x2])

    end

    @testset "gpfit(model, initialvalues)" begin
        # non-stationary location, logscale and shape
        model = Extremes.ThresholdExceedance(Variable("y", y),
            logscalecov = [x₁],
            shapecov = [x₂])

        fm = Extremes.gpfit(model, zeros(Float64, 4))

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
