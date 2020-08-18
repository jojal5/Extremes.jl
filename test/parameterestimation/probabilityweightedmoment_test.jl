@testset "probabilityweightedmoment.jl" begin
    @testset "pwm(x, p, r, s)" begin

        # p < 0 throws error
        @test_throws AssertionError Extremes.pwm([1.0], -1, 1, 1)

        # r < 0 throws error
        @test_throws AssertionError Extremes.pwm([1.0], 1, -1, 1)

        #  s < 0 throws error
        @test_throws AssertionError Extremes.pwm([1.0], 1, 1, -1)

        # Probability weighted moment with p=1, r=0 and s=k
        μ = .5
        σ = 1.0
        pd = Logistic(μ,σ)
        y = rand(pd, 10000)
        # theoritical form of the pwm for the Logistic distribution (Greenwood and al., 1979)
        f(k::Int) = μ/(1+k) - σ/(1+k)*sum(1/i for i=1:k)

        for k=1:3
            @test f(k) ≈ Extremes.pwm(y,1,0,k) rtol=.1
        end

        # Probability weighted moment with p=1, r=j and s=0
        μ = .5
        σ = 1.0
        ξ = 0
        pd = GeneralizedExtremeValue(μ,σ,0)
        y = rand(pd, 10000)

        # theoritical form of the pwm for the Gumbel distribution (Greenwood and al., 1979)
        f(j::Int) = μ/(1+j) + σ/(1+j) * (log(1+j) + MathConstants.eulergamma)

        for j=1:3
            @test f(j) ≈ Extremes.pwm(y,1,j,0) rtol=.1
        end

    end

    include(joinpath("probabilityweightedmoment", "probabilityweightedmoment_gev_test.jl"))
    include(joinpath("probabilityweightedmoment", "probabilityweightedmoment_gp_test.jl"))
    include(joinpath("probabilityweightedmoment", "probabilityweightedmoment_gumbel_test.jl"))

end
