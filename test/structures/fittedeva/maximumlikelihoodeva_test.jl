@testset "maximumlikelihoodeva.jl" begin
    n = 1000
    θ = [0.0, 1.0, 0.1]

    pd = GeneralizedExtremeValue(θ...)
    y = rand(pd, n)

    bm_model = Extremes.MaximumLikelihoodEVA(Extremes.BlockMaxima(Variable("y", y)), θ)

    @testset "hessian(model)" begin
        # TODO : Test with known values (J)

    end

    @testset "parametervar(fm)" begin
        # TODO : Test with known values (J)

    end

    @testset "loglike(fd)" begin
        # TODO : Test with known values (J)

    end

    @testset "getdistribution(fittedmodel)" begin
        # stationary
        n = 100

        μ = 0.0
        σ = 1.0
        ξ = 0.1
        ϕ = log(σ)

        θ = [μ; ϕ; ξ]

        pd = GeneralizedExtremeValue(μ, σ, ξ)
        y = rand(pd, n)

        model = MaximumLikelihoodEVA(BlockMaxima(Variable("y", y)), θ)

        fd = Extremes.getdistribution(model)[]

        @test fd == pd

    end

    @testset "quantile(fm, p)" begin
        # p not in [0, 1] throws
        @test_throws AssertionError Extremes.quantile(bm_model, -1)

        # TODO : Test with known values

    end

    @testset "quantilevar(fm, level)" begin
        # TODO : Test with known values (J)

    end

    @testset "returnlevel(fm, returnPeriod, confidencelevel)" begin
        # returnPeriod < 0 throws
        @test_throws AssertionError Extremes.returnlevel(bm_model, -1, 0.95)

        # confidencelevel not in [0, 1] throws
        @test_throws AssertionError Extremes.returnlevel(bm_model, 1, -1)

        # TODO : Test with known values (J)

    end

    @testset "returnlevel(fm, threshold, nobservation, nobsperblock, returnPeriod, confidencelevel)" begin
        n = 1000
        θ = [0.0, 1.0, 0.1]

        pd = GeneralizedExtremeValue(θ...)
        y = rand(pd, n)

        te_model = Extremes.MaximumLikelihoodEVA(Extremes.ThresholdExceedance(Variable("y", y)), θ)

        # returnPeriod < 0 throws
        @test_throws AssertionError Extremes.returnlevel(te_model, 0, n, 1, -1, 0.95)

        # confidencelevel not in [0, 1] throws
        @test_throws AssertionError Extremes.returnlevel(te_model, 0, n, 1, 1, -1)

        # TODO : Test with known values (J)
    end

    @testset "Base.show(io, obj)" begin
        # print does not throw
        buffer = IOBuffer()
        @test_logs Base.show(buffer, bm_model)

    end

end
