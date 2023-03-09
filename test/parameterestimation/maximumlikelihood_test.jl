@testset "maximumlikelihood.jl" begin

    @testset "fit(model, initialvalues) -- arguments control" begin
        
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

    @testset "fit(model) no solution" begin
        # No solution warn test
		y = [14.6, -0.5, 505.9]

        model = Extremes.BlockMaxima{GeneralizedExtremeValue}(Variable("y", y))

        @test_logs (:warn,"The maximum likelihood algorithm did not find a solution. Maybe try with different initial values or with another method. The returned values are the initial values.") Extremes.fit(model)

    end
       
    @test_set "fit(BlockMaxima{GeneralizedExtremeValue}, initialvalues) -- stationary" begin

        df = CSV.read("dataset/gev_stationary.csv", DataFrame)

        model = Extremes.BlockMaxima{GeneralizedExtremeValue}(Variable("y", df.y))

        fm = Extremes.fit(model, [0., 0., 0.])

        @test all(isapprox.(fm.θ̂,[0.0009, 0.0142, -0.0060], atol=.0001))

    end

    @test_set "fit(BlockMaxima{GeneralizedExtremeValue}, initialvalues) -- non-stationary" begin

        df = CSV.read("dataset/gev_nonstationary.csv", DataFrame)

        model = Extremes.BlockMaxima{GeneralizedExtremeValue}(Variable("y", df.y),
            locationcov = [Variable("x₁", df.x₁)],
            logscalecov = [Variable("x₂", df.x₂)],
            shapecov = [Variable("x₃", df.x₃)])

        fm = Extremes.fit(model, [0., 0., 0., 0., 0., 0.])

        @test all(isapprox.(fm.θ̂, [1.0182, 1.0036, -0.4793, 0.9833, -0.0093, -0.0451], atol=.0001))

    end

    @test_set "fit(BlockMaxima{Gumbel}, initialvalues) -- stationary" begin

        df = CSV.read("dataset/gev_stationary.csv", DataFrame)

        model = Extremes.BlockMaxima{Gumbel}(Variable("y", df.y))

        fm = Extremes.fit(model, [0., 0.])

        @test all(isapprox.(fm.θ̂, [-0.0023, 0.0124], atol=.0001))

    end

    @test_set "fit(BlockMaxima{Gumbel}, initialvalues) -- non-stationary" begin

        df = CSV.read("dataset/gev_nonstationary.csv", DataFrame)

        model = Extremes.BlockMaxima{Gumbel}(Variable("y", df.y),
            locationcov = [Variable("x₁", df.x₁)],
            logscalecov = [Variable("x₂", df.x₂)])

        fm = Extremes.fit(model, [0., 0., 0., 0.])

        @test all(isapprox.(fm.θ̂, [1.0155, 1.0035, -0.4820, 0.9847], atol=.0001))

    end

    @test_set "fit(ThresholdExceedance, initialvalues) -- stationary" begin

        df = CSV.read("test/dataset/gp_stationary.csv", DataFrame)

        model = Extremes.ThresholdExceedance(Variable("y", df.y))

        fm = Extremes.fit(model, [0., 0.])

        @test all(isapprox.(fm.θ̂, [-0.0135, 0.0059], atol=.0001))

    end

    @test_set "fit(ThresholdExceedance, initialvalues) -- non-stationary" begin

        df = CSV.read("dataset/gp_nonstationary.csv", DataFrame)

        model = Extremes.ThresholdExceedance(Variable("y", df.y),
            logscalecov = [Variable("x₁", df.x₁)])

        fm = Extremes.fit(model, [0., 0., 0.])

        @test all(isapprox.(fm.θ̂, [-0.4957, 1.0136, -0.0034], atol=.0001))

    end

    include(joinpath("maximumlikelihood", "maximumlikelihood_gev_test.jl"))
    include(joinpath("maximumlikelihood", "maximumlikelihood_gp_test.jl"))

end
