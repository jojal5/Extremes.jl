@testset "bayesianeva.jl" begin


    μ, σ, ξ = 100.0, 5.0, 0.1

    pd = GeneralizedExtremeValue(μ, σ, ξ)
    y = [100.0]

    fm = Extremes.BayesianEVA(Extremes.BlockMaxima(Variable("y", y)), Mamba.Chains([100.0 log(5.0) .1]))


    @testset "quantile(fm, p)" begin
        # p not in [0, 1] throws
        @test_throws AssertionError Extremes.quantile(fm, -1)

        # Test with known values
        @test quantile(fm, .95)[] ≈ quantile(pd, .95)

    end

    @testset "returnlevel(fm, returnPeriod, confidencelevel)" begin
        # returnPeriod < 0 throws
        @test_throws AssertionError Extremes.returnlevel(fm, -1, 0.95)

        # confidencelevel not in [0, 1]
        @test_throws AssertionError Extremes.returnlevel(fm, 1, -1)

        # Test with known values
        @test returnlevel(fm, 100, .95).value[] ≈ quantile(pd, 1-1/100)

    end

    @testset "returnlevel(fm, threshold, nobservation, nobsperblock, returnPeriod, confidencelevel)" begin
        # TODO : Test when implemented

    end

    @testset "showfittedEVA(io, obj, prefix)" begin
        # print does not throw
        buffer = IOBuffer()
        @test_logs Extremes.showfittedEVA(buffer, fm, prefix = "\t")
    end

    @testset "showChain(io, chain, prefix)" begin
        # print does not throw
        buffer = IOBuffer()
        @test_logs Extremes.showChain(buffer, fm.sim)
    end

    @testset "cint(fm::BayesianEVA)" begin

        μ, ϕ, ξ = 100, log(5), .1
        y = rand(GeneralizedExtremeValue(μ,exp(ϕ), ξ), 1000)

        fm = gevfitbayes(y)

        @test_throws AssertionError cint(fm, 1.95)
        @test_throws AssertionError cint(fm, -1.95)

        confint = cint(fm)

        @test confint[1][1] < μ < confint[1][2]
        @test confint[2][1] < ϕ < confint[2][2]
        @test confint[3][1] < ξ < confint[3][2]

        confint = cint(fm, .99)

        @test confint[1][1] < μ < confint[1][2]
        @test confint[2][1] < ϕ < confint[2][2]
        @test confint[3][1] < ξ < confint[3][2]

    end

end
