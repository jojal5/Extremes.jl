@testset "maximumlikelihood.jl" begin
    @testset "fit(model)" begin
        # TODO : Add no solution warn test

        # stationary GEV fit by ML
        n = 10000

        μ = 0.0
        σ = 1.0
        ξ = 0.1

        ϕ = log(σ)
        θ = [μ; ϕ; ξ]

        pd = GeneralizedExtremeValue(μ, σ, ξ)
        y = rand(pd, n)

        model = Extremes.BlockMaxima(y)

        fm = Extremes.fit(model)

        @test fm.θ̂ ≈ θ atol = 0.05

        # non-stationary location GEV fit by ML
        n = 10000

        x₁ = randn(n)
        x₂ = randn(n)

        μ = x₁ + x₂
        σ = 1.0
        ξ = 0.1

        ϕ = log(σ)
        θ = [0.0; 1.0; 1.0; ϕ; ξ]

        pd = GeneralizedExtremeValue.(μ, σ, ξ)
        y = rand.(pd)

        model = Extremes.BlockMaxima(y, locationcov = [ExplanatoryVariable("x₁", x₁), ExplanatoryVariable("x₂", x₂)])

        fm = Extremes.fit(model)

        @test fm.θ̂ ≈ θ atol = 0.05

        # non-stationary logscale GEV fit by ML
        n = 10000

        x₁ = randn(n) / 3
        x₂ = randn(n) / 3

        μ = 0.0
        ϕ = x₁ + x₂
        ξ = 0.1

        σ = exp.(ϕ)
        θ = [μ; 0.0; 1.0; 1.0; ξ]

        pd = GeneralizedExtremeValue.(μ, σ, ξ)
        y = rand.(pd)

        model = Extremes.BlockMaxima(y, logscalecov = [ExplanatoryVariable("x₁", x₁), ExplanatoryVariable("x₂", x₂)])

        fm = Extremes.fit(model)

        @test fm.θ̂ ≈ θ atol = 0.05

        # non-stationary location and logscale GEV fit by ML
        n = 10000

        x₁ = randn(n)
        x₂ = randn(n) / 3

        μ = 1.0 .+ x₁
        ϕ = -.05 .+ x₂
        ξ = 0.1

        σ = exp.(ϕ)
        θ = [1.0; 1.0; -.05; 1.0; ξ]

        pd = GeneralizedExtremeValue.(μ, σ, ξ)
        y = rand.(pd)

        model = Extremes.BlockMaxima(y, locationcov = [ExplanatoryVariable("x₁", x₁)], logscalecov = [ExplanatoryVariable("x₂", x₂)])

        fm = Extremes.fit(model)

        @test fm.θ̂ ≈ θ atol = 0.05

        # non-stationary shape GEV fit by ML
        n = 10000

        x₁ = randn(n) / 10

        μ = 0.0
        ϕ = 0.0
        ξ = x₁

        σ = exp(ϕ)
        θ = [μ; ϕ; 0.0; 1.0]

        pd = GeneralizedExtremeValue.(μ, σ, ξ)
        y = rand.(pd)

        model = Extremes.BlockMaxima(y, shapecov = [ExplanatoryVariable("x₁", x₁)])

        fm = Extremes.fit(model)

        @test fm.θ̂ ≈ θ atol = 0.1

        # stationary GP fit by ML
        n = 10000

        σ = 1.0
        ξ = 0.1

        ϕ = log(σ)
        θ = [ϕ; ξ]

        pd = GeneralizedPareto(σ, ξ)
        y = rand(pd, n)

        model = Extremes.ThresholdExceedance(y)

        fm = Extremes.fit(model)

        @test fm.θ̂ ≈ θ atol = 0.05

        # non-stationary logscale GP fit by ML
        n = 10000

        x₁ = randn(n) / 3
        x₂ = randn(n) / 3

        ϕ = -.5 .+ x₁ .+ x₂
        ξ = 0.1

        σ = exp.(ϕ)
        θ = [-.5; 1.0; 1.0; ξ]

        pd = GeneralizedPareto.(σ, ξ)
        y = rand.(pd)

        model = Extremes.ThresholdExceedance(y, logscalecov = [ExplanatoryVariable("x₁", x₁), ExplanatoryVariable("x₂", x₂)])

        fm = Extremes.fit(model)

        @test fm.θ̂ ≈ θ atol = 0.05

        # non-stationary shape GP fit by ML
        n = 10000

        x₁ = randn(n) / 10

        μ = 0.0
        ϕ = 0.0
        ξ = x₁

        σ = exp(ϕ)
        θ = [ϕ; 0.0; 1.0]

        pd = GeneralizedPareto.(μ, σ, ξ)
        y = rand.(pd)

        model = Extremes.ThresholdExceedance(y, shapecov = [ExplanatoryVariable("x₁", x₁)])

        fm = Extremes.fit(model)

        @test fm.θ̂ ≈ θ atol = 0.1

    end

    include(joinpath("maximumlikelihood", "maximumlikelihood_gev_test.jl"))
    include(joinpath("maximumlikelihood", "maximumlikelihood_gp_test.jl"))

end
