@testset "pwmeva.jl" begin
    n = 1000
    θ = [0.0, 1.0, 0.1]

    pd = GeneralizedExtremeValue(θ...)
    y = rand(pd, n)

    pwm_model = Extremes.pwmEVA(Extremes.BlockMaxima(Variable("y", y)), θ)

    @testset "quantile(fm, p)" begin
        # p outside of [0, 1] throws
        @test_throws AssertionError Extremes.quantile(pwm_model, -1)

        # TODO : test with known values (J)

    end

    @testset "parametervar(fm, nboot)" begin
        # nboot < 0 throws
        @test_throws AssertionError Extremes.parametervar(pwm_model, -1)

        # TODO : test with known values (J)

    end

    @testset "quantilevar(fm, level, nboot)" begin
        # TODO : test with known values (J)

    end

    @testset "Base.show(io, obj)" begin
        # print does not throw
        buffer = IOBuffer()
        @test_logs Base.show(buffer, pwm_model)

    end

end
