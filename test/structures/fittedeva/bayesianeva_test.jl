@testset "bayesianeva.jl" begin
    n = 1000

    pd = GeneralizedExtremeValue(0.0, 1.0, 0.1)
    y = rand(pd, n)

    bm_model = Extremes.BayesianEVA(Extremes.BlockMaxima(Variable("y", y)), Mamba.Chains(100, 3))

    @testset "quantile(fm, p)" begin
        # p not in [0, 1] throws
        @test_throws AssertionError Extremes.quantile(bm_model, -1)

        # TODO : Test with known values (J)
        #        If creating a Mamba.Chains is too complicated, fitbayes could be
        #        called once for the whole bayesianeva.jl testset.

    end

    @testset "returnlevel(fm, returnPeriod, confidencelevel)" begin
        # returnPeriod < 0 throws
        @test_throws AssertionError Extremes.returnlevel(bm_model, -1, 0.95)

        # confidencelevel not in [0, 1]
        @test_throws AssertionError Extremes.returnlevel(bm_model, 1, -1)

        # TODO : Test with known values (J)

    end

    @testset "returnlevel(fm, threshold, nobservation, nobsperblock, returnPeriod, confidencelevel)" begin
        # TODO : Test when implemented

    end

    @testset "Base.show(io, obj)" begin
        # print does not throw
        buffer = IOBuffer()
        @test_logs Base.show(buffer, bm_model)
    end

end
