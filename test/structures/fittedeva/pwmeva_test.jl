@testset "pwmeva.jl" begin

    θ = [0.0, 1.0, 0.1]

    pd = GeneralizedExtremeValue(θ...)

    y = [0]

    fm = Extremes.pwmEVA{BlockMaxima, GeneralizedExtremeValue}(Extremes.BlockMaxima(Variable("y", y)), [θ[1]; log(θ[2]); θ[3]])

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

    @testset "returnlevel_cint(fm, returnPeriod, confidencelevel)" begin
        pd = GeneralizedExtremeValue(10,1,.1)
        y = rand(pd,1000)

        fm = gevfitpwm(y)

        # returnPeriod < 0 throws
        @test_throws AssertionError returnlevel_cint(fm, -1, 0.95)

        # confidencelevel not in [0, 1]
        @test_throws AssertionError returnlevel_cint(fm, 1, -1)

        r = returnlevel_cint(fm, 100, .95)
        q = quantile(pd, 1-1/100)

        # Test with known values
        @test r.value[] ≈ q rtol = .05
        @test r.cint[][1] < q < r.cint[][2]
    end

    @testset "returnlevel_cint(fm, returnPeriod, confidencelevel)" begin
        threshold = 10.0
        pd = GeneralizedPareto(threshold, 1,.1)
        y = rand(pd,1000) .- threshold

        fm = gpfitpwm(y)

        # returnPeriod < 0 throws
        @test_throws AssertionError returnlevel_cint(fm, threshold, length(y), 1, -1, 0.95)

        # confidencelevel not in [0, 1]
        @test_throws AssertionError returnlevel_cint(fm, threshold,length(y), 1, 100, 1.95)

        # Test with known values
        r = returnlevel_cint(fm, threshold, length(y), 1, 100, .95)
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


@testset "cint(fm::pwmEVA)" begin

    μ, ϕ, ξ = 100, log(5), .1
    y = rand(GeneralizedExtremeValue(μ,exp(ϕ), ξ), 1000)

    fm = gevfitpwm(y)

    @test_throws AssertionError cint(fm, 1.95)
    @test_throws AssertionError cint(fm, -1.95)
    @test_throws AssertionError cint(fm, .95, -10)

    confint = cint(fm)

    @test confint[1][1] < μ < confint[1][2]
    @test confint[2][1] < ϕ < confint[2][2]
    @test confint[3][1] < ξ < confint[3][2]

    confint = cint(fm, .99)

    @test confint[1][1] < μ < confint[1][2]
    @test confint[2][1] < ϕ < confint[2][2]
    @test confint[3][1] < ξ < confint[3][2]

    confint = cint(fm, .99, 1000)

    @test confint[1][1] < μ < confint[1][2]
    @test confint[2][1] < ϕ < confint[2][2]
    @test confint[3][1] < ξ < confint[3][2]

end
