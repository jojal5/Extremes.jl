@testset "thresholdexceedance.jl" begin
    @testset "ThresholdExceedance(exceedances; logscalecov, shapecov)" begin

    end

    @testset "paramindex(model)" begin

    end

    @testset "getcovariatenumber(model)" begin

    end

    @testset "nparameter(model)" begin

    end

    @testset "getdistribution(model, θ)" begin
        # stationary
        n = 100

        σ = 1.0
        ξ = .1
        ϕ = log(σ)

        θ = [ϕ ; ξ]

        pd = GeneralizedPareto(σ, ξ)
        y = rand(pd,n)

        model = ThresholdExceedance(y)

        fd = Extremes.getdistribution(model, θ)[]

        @test fd == pd

        # non-stationary
        n = 100

        x₁ = randn(n)/3
        x₂ = randn(n)/3
        x₃ = randn(n)/10

        θ = [-.5 ; 1.0 ; 1.0 ; 0 ; 1.0]

        ϕ = θ[1] .+ θ[2] * x₁ .+ θ[3] * x₂
        ξ = θ[4] .+ θ[5] * x₃

        σ = exp.(ϕ)

        pd = GeneralizedPareto.(σ, ξ)
        y = rand.(pd)

        model = ThresholdExceedance(y, logscalecov = [ExplanatoryVariable("x₁", x₁), ExplanatoryVariable("x₂", x₂)], shapecov = [ExplanatoryVariable("x₃", x₃)])

        fd = Extremes.getdistribution(model, θ)

        @test fd == pd

    end

    @testset "getinitialvalue(::Type{GeneralizedPareto},y)" begin

    end

    @testset "getinitialvalue(model)" begin

    end

    @testset "showEVA(io, obj; prefix)" begin

    end

end
