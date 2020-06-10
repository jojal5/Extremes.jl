@testset "blockmaxima.jl" begin
    @testset "BlockMaxima(data; locationcov, logscalecov, shapecov)" begin

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

        μ = 0.0
        σ = 1.0
        ξ = 0.1
        ϕ = log(σ)

        θ = [μ; ϕ; ξ]

        pd = GeneralizedExtremeValue(μ, σ, ξ)
        y = rand(pd, n)

        model = BlockMaxima(y)

        fd = Extremes.getdistribution(model, θ)[]

        @test fd == pd

        # non-stationary
        n = 10000

        x₁ = randn(n)
        x₂ = randn(n)/3
        x₃ = randn(n)/10

        θ = [5.0 ; 1.0 ; -.5 ; 1.0 ; 0.0 ; 1.0]

        μ = θ[1] .+ θ[2] * x₁
        ϕ = θ[3] .+ θ[4] * x₂
        ξ = θ[5] .+ θ[6] * x₃

        pd = GeneralizedExtremeValue.(μ, exp.(ϕ), ξ)

        y = rand.(pd)

        model = BlockMaxima(y, locationcov = [ExplanatoryVariable("x₁", x₁)], logscalecov = [ExplanatoryVariable("x₂", x₂)], shapecov = [ExplanatoryVariable("x₃", x₃)])

        fd = Extremes.getdistribution(model, θ)

        @test pd == fd

    end

    @testset "getinitialvalue(::Type{GeneralizedExtremeValue},y)" begin

    end

    @testset "getinitialvalue(model)" begin

    end

    @testset "showEVA(io, obj; prefix)" begin

    end

end
