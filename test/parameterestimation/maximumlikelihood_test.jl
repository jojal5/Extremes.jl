@testset "maximumlikelihood.jl" begin
    @testset "fit(model, initialvalues)" begin
        
        df = CSV.read("dataset/gev_nonstationary.csv", DataFrame)

        deleteat!(df, 101:nrow(df))

        model = Extremes.BlockMaxima{GeneralizedExtremeValue}(Variable("y", df.y))

        # Initial value vector length != nparameter throws
        @test_throws AssertionError Extremes.fit(model, [0.0, 0.0, 0.0, 0.0])

        # Initial value vector invalid throws
        @test_throws AssertionError Extremes.fit(model, [Inf, Inf, Inf])

        # Initial value vector valid and length == nparameter does not throw
        @test_logs Extremes.fit(model, [0.0, 0.0, 0.0])

    end

    @testset "fit(model)" begin
        # No solution warn test
		y = [14.6, -0.5, 505.9]

        model = Extremes.BlockMaxima{GeneralizedExtremeValue}(Variable("y", y))

        @test_logs (:warn,"The maximum likelihood algorithm did not find a solution. Maybe try with different initial values or with another method. The returned values are the initial values.") Extremes.fit(model)

        # stationary GEV fit by ML

        df = CSV.read("dataset/gev_stationary.csv", DataFrame)

        model = Extremes.BlockMaxima{GeneralizedExtremeValue}(Variable("y", df.y))

        fm = Extremes.gevfit(model, [0., 0., 0.])

        @test all(isapprox.(fm.θ̂,[0.0009, 0.0142, -0.0060], atol=.0001))

        # non-stationary GEV fit by ML
        df = CSV.read("dataset/gev_nonstationary.csv", DataFrame)

        model = Extremes.BlockMaxima{GeneralizedExtremeValue}(Variable("y", df.y),
            locationcov = [Variable("x₁", df.x₁)],
            logscalecov = [Variable("x₂", df.x₂)],
            shapecov = [Variable("x₃", df.x₃)])

        fm = Extremes.gevfit(model, [0., 0., 0., 0., 0., 0.])

        @test all(isapprox.(fm.θ̂,[1.0182, 1.0036, -0.4793, 0.9833, -0.0093, -0.0451], atol=.0001))

        # stationary GP fit by ML
        n = 5000

        σ = 1.0
        ξ = 0.1

        ϕ = log(σ)
        θ = [ϕ; ξ]

        pd = GeneralizedPareto(σ, ξ)
        y = rand(pd, n)

        model = Extremes.ThresholdExceedance(Variable("y", y))

        fm = Extremes.fit(model)

        cinterval = cint(fm)

        @test [x[1] for x in cinterval] <= θ
        @test θ <= [x[2] for x in cinterval]

        # non-stationary GP fit by ML
        n = 5000

        x₁ = randn(n) / 3
        x₂ = randn(n) / 10

        ϕ = -.5 .+ x₁
        ξ = x₂

        σ = exp.(ϕ)
        θ = [-.5; 1.0; 0.0; 1.0]

        pd = GeneralizedPareto.(σ, ξ)
        y = rand.(pd)

        model = Extremes.ThresholdExceedance(Variable("y", y), logscalecov = [Variable("x₁", x₁)], shapecov = [Variable("x₂", x₂)])

        fm = Extremes.fit(model)

        cinterval = cint(fm)

        @test [x[1] for x in cinterval] <= θ
        @test θ <= [x[2] for x in cinterval]

    end

    include(joinpath("maximumlikelihood", "maximumlikelihood_gev_test.jl"))
    include(joinpath("maximumlikelihood", "maximumlikelihood_gp_test.jl"))

end
