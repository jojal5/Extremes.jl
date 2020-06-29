@testset "pwmeva.jl" begin

    θ = [0.0, 1.0, 0.1]

    pd = GeneralizedExtremeValue(θ...)

    y = [0]

    fm = Extremes.pwmEVA(Extremes.BlockMaxima(Variable("y", y)), [θ[1]; log(θ[2]); θ[3]])

    @testset "quantile(fm, p)" begin
        # p outside of [0, 1] throws
        @test_throws AssertionError Extremes.quantile(fm, -1)

        # Test with known values
        @test quantile(fm, .99)[] ≈ quantile(pd,.99)

    end

    @testset "parametervar(fm, nboot)" begin
        # nboot < 0 throws
        @test_throws AssertionError Extremes.parametervar(fm, -1)

        # TODO : test with known values (J)

    end

    @testset "quantilevar(fm, level, nboot)" begin
        # TODO : test with known values (J)

    end

    @testset "returnlevel(fm, returnPeriod, confidencelevel)" begin
        pd = GeneralizedExtremeValue(10,1,.1)
        y = rand(pd,1000)

        fm = gevfitpwm(y)

        # returnPeriod < 0 throws
        @test_throws AssertionError returnlevel(fm, -1, 0.95)

        # confidencelevel not in [0, 1]
        @test_throws AssertionError returnlevel(fm, 1, -1)

        r = returnlevel(fm, 100, .95)
        q = quantile(pd, 1-1/100)

        # Test with known values
        @test r.value[] ≈ q rtol = .05
        @test r.cint[][1] < q < r.cint[][2]
    end

    @testset "returnlevel(fm, returnPeriod, confidencelevel)" begin
        threshold = 10.0
        pd = GeneralizedPareto(threshold, 1,.1)
        y = rand(pd,1000) .- threshold

        fm = gpfitpwm(y)

        # returnPeriod < 0 throws
        @test_throws AssertionError returnlevel(fm, threshold, length(y), 1, -1, 0.95)

        # confidencelevel not in [0, 1]
        @test_throws AssertionError returnlevel(fm, threshold,length(y), 1, 100, 1.95)

        # Test with known values
        r = returnlevel(fm, threshold, length(y), 1, 100, .95)
        q = quantile(pd, 1-1/100)

        # Test with known values
        @test r.value[] ≈ q rtol = .05
        @test r.cint[][1] < q < r.cint[][2]
    end


    @testset "showfittedEVA(io, obj, prefix)" begin
        # print does not throw
        buffer = IOBuffer()
        @test_logs Extremes.showfittedEVA(buffer, fm, prefix = "\t")

    end

end
