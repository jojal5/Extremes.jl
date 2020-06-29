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

        threshold = 10.0

        x = Variable("x",collect(range(0, stop=1, length=1000)))
        ϕ = x.value
        σ = exp.(ϕ)

        pd = GeneralizedPareto.(threshold, σ, .1)

        y = rand.(pd) .- threshold

        fm = gpfitbayes(y, logscalecov = [x])

        # returnPeriod < 0 throws
        @test_throws AssertionError Extremes.returnlevel(fm, -1, 0.95)

        # confidencelevel not in [0, 1]
        @test_throws AssertionError Extremes.returnlevel(fm, 1, -1)

        # Test with known values
        r = returnlevel(fm, threshold, length(y), 1, 100, .95)
        q = quantile.(pd, 1-1/100)

        @test r.cint[1][1] < q[1] < r.cint[1][2]  # Beginning of interval
        @test r.cint[end][1] < q[end] < r.cint[end][2]  # End of interval

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

end
