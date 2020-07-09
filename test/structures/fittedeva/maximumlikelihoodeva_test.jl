@testset "maximumlikelihoodeva.jl" begin

    x₁ = Variable("x₁", [1])
    x₂ = Variable("x₂", [2])
    x₃ = Variable("x₂", [3])

    μ = 1 + x₂.value[] + x₃.value[]
    ϕ = -.5 + x₁.value[]
    ξ = .1

    pd = GeneralizedExtremeValue(μ, exp(ϕ), ξ)

    model = BlockMaxima(Variable("y", [6]), locationcov=[x₂; x₃], logscalecov = [x₁])

    fm = Extremes.MaximumLikelihoodEVA(model, [1; 1; 1; -.5; 1; .1])


    @testset "hessian(model)" begin
        # TODO : Test with known values

    end

    @testset "parametervar(fm)" begin
        # TODO : Test with known values

    end

    @testset "loglike(fd)" begin
        # Test with known values
        @test Extremes.loglike(fm) ≈ logpdf(pd,6)

    end

    @testset "getdistribution(fittedmodel)" begin
        @test Extremes.getdistribution(fm)[] == pd
    end

    @testset "quantile(fm, p)" begin
        # p not in [0, 1] throws
        @test_throws AssertionError Extremes.quantile(fm, -1)

        # Test with known values
        @test quantile(fm, .99)[] ≈ quantile(pd,.99)

    end

    @testset "quantilevar(fm, level)" begin
        # TODO : Test with known values (J)

    end

    @testset "returnlevel(fm, returnPeriod)" begin
        # returnPeriod < 0 throws
        @test_throws AssertionError Extremes.returnlevel(fm, -1)

        # TODO: Test with known values

    end

    @testset "cint(fm, returnPeriod, confidencelevel)" begin
        # returnPeriod < 0 throws
        @test_throws AssertionError Extremes.cint(ReturnLevel(Extremes.BlockMaximaModel(fm), -1, [1.0]), 0.95)

        # confidencelevel not in [0, 1] throws
        @test_throws AssertionError Extremes.cint(ReturnLevel(Extremes.BlockMaximaModel(fm), 1, [1.0]), -1)

        # TODO: Test with known values

    end

    @testset "returnlevel(fm, threshold, nobservation, nobsperblock, returnPeriod)" begin
        n = 1000
        θ = [0.0, 1.0, 0.1]

        pd = GeneralizedExtremeValue(θ...)
        y = rand(pd, n)

        te_model = Extremes.MaximumLikelihoodEVA(Extremes.ThresholdExceedance(Variable("y", y)), θ)

        # returnPeriod < 0 throws
        @test_throws AssertionError Extremes.returnlevel(te_model, 0, n, 1, -1)

        # TODO : Test with known values (J)
    end

    @testset "cint(fm, threshold, nobservation, nobsperblock, returnPeriod, confidencelevel)" begin
        n = 1000
        θ = [0.0, 1.0, 0.1]

        pd = GeneralizedExtremeValue(θ...)
        y = rand(pd, n)

        te_model = Extremes.MaximumLikelihoodEVA(Extremes.ThresholdExceedance(Variable("y", y)), θ)

        # returnPeriod < 0 throws
        @test_throws AssertionError Extremes.cint(ReturnLevel(Extremes.PeakOverThreshold(te_model, 0, n, 1), -1, [1.0]), 0, n, 1, 0.95)

        # confidencelevel not in [0, 1] throws
        @test_throws AssertionError Extremes.cint(ReturnLevel(Extremes.PeakOverThreshold(te_model, 0, n, 1), 1, [1.0]), 0, n, 1, -1)

        # TODO : Test with known values (J)
    end

    @testset "showfittedEVA(io, obj, prefix)" begin
        # print does not throw
        buffer = IOBuffer()
        @test_logs Extremes.showfittedEVA(buffer, fm, prefix = "\t")

    end

    @testset "cint(fm::MaximumLikelihoodEVA)" begin

        data = load("portpirie")
        fm = gevfit(data, :SeaLevel)

        @test_throws AssertionError cint(fm, 1.95)
        @test_throws AssertionError cint(fm, -1.95)

        confint = cint(fm)

        @test confint[1] ≈ [3.82; 3.93] atol = .02
        @test confint[2] ≈ log.([0.158,0.238]) atol = .05
        @test confint[3] ≈ [-0.242; 0.142] atol = .02

        confint = cint(fm, .95)

        @test confint[1] ≈ [3.82; 3.93] atol = .02
        @test confint[2] ≈ log.([0.158,0.238]) atol = .05
        @test confint[3] ≈ [-0.242; 0.142] atol = .02



    end

end
