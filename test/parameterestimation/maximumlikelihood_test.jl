@testset "maximumlikelihood.jl" begin
    @testset "fit(model)" begin

        # No solution warn test
        n = 10

        μ = 0.0
        σ = 1.0
        ξ = .8

        ϕ = log(σ)
        θ = [μ; ϕ; ξ]

        pd = GeneralizedExtremeValue(μ, σ, ξ)

        y = rand(pd, n)

        @test_logs (:warn,"The maximum likelihood algorithm did not find a solution. Maybe try with different initial values or with another method. The returned values are the initial values.") gevfit(y)

        # stationary GEV fit by ML
        n = 5000

        μ = 0.0
        σ = 1.0
        ξ = 0.1

        ϕ = log(σ)
        θ = [μ; ϕ; ξ]

        pd = GeneralizedExtremeValue(μ, σ, ξ)
        y = rand(pd, n)

        model = Extremes.BlockMaxima(y)

        fm = Extremes.fit(model)

        varM = Extremes.parametervar(fm)
        var = sqrt.([varM[i,i] for i in 1:length(θ)]) .* quantile(Normal(), 0.975)

        @test fm.θ̂ .- var <= θ
        @test θ <= fm.θ̂ .+ var

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

        model = Extremes.BlockMaxima(y, locationcov = [Variable("x₁", x₁), Variable("x₂", x₂)])

        fm = Extremes.fit(model)

        varM = Extremes.parametervar(fm)
        var = sqrt.([varM[i,i] for i in 1:length(θ)]) .* quantile(Normal(), 0.975)

        @test fm.θ̂ .- var <= θ
        @test θ <= fm.θ̂ .+ var

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

        model = Extremes.BlockMaxima(y, logscalecov = [Variable("x₁", x₁), Variable("x₂", x₂)])

        fm = Extremes.fit(model)

        varM = Extremes.parametervar(fm)
        var = sqrt.([varM[i,i] for i in 1:length(θ)]) .* quantile(Normal(), 0.975)

        @test fm.θ̂ .- var <= θ
        @test θ <= fm.θ̂ .+ var

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

        model = Extremes.BlockMaxima(y, locationcov = [Variable("x₁", x₁)], logscalecov = [Variable("x₂", x₂)])

        fm = Extremes.fit(model)

        varM = Extremes.parametervar(fm)
        var = sqrt.([varM[i,i] for i in 1:length(θ)]) .* quantile(Normal(), 0.975)

        @test fm.θ̂ .- var <= θ
        @test θ <= fm.θ̂ .+ var

        # # non-stationary shape GEV fit by ML
        # n = 10000
        #
        # x₁ = randn(n) / 10
        #
        # μ = 0.0
        # ϕ = 0.0
        # ξ = x₁
        #
        # σ = exp(ϕ)
        # θ = [μ; ϕ; 0.0; 1.0]
        #
        # pd = GeneralizedExtremeValue.(μ, σ, ξ)
        # y = rand.(pd)
        #
        # model = Extremes.BlockMaxima(y, shapecov = [Variable("x₁", x₁)])
        #
        # fm = Extremes.fit(model)
        #
        #varM = Extremes.parametervar(fm)
        #var = sqrt.([varM[i,i] for i in 1:length(θ)]) .* quantile(Normal(), 0.975)

        #@test fm.θ̂ .- var <= θ
        #@test θ <= fm.θ̂ .+ var

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

        varM = Extremes.parametervar(fm)
        var = sqrt.([varM[i,i] for i in 1:length(θ)]) .* quantile(Normal(), 0.975)

        @test fm.θ̂ .- var <= θ
        @test θ <= fm.θ̂ .+ var

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

        model = Extremes.ThresholdExceedance(y, logscalecov = [Variable("x₁", x₁), Variable("x₂", x₂)])

        fm = Extremes.fit(model)

        varM = Extremes.parametervar(fm)
        var = sqrt.([varM[i,i] for i in 1:length(θ)]) .* quantile(Normal(), 0.975)

        @test fm.θ̂ .- var <= θ
        @test θ <= fm.θ̂ .+ var

        # # non-stationary shape GP fit by ML
        # n = 10000
        #
        # x₁ = randn(n) / 10
        #
        # μ = 0.0
        # ϕ = 0.0
        # ξ = x₁
        #
        # σ = exp(ϕ)
        # θ = [ϕ; 0.0; 1.0]
        #
        # pd = GeneralizedPareto.(μ, σ, ξ)
        # y = rand.(pd)
        #
        # model = Extremes.ThresholdExceedance(y, shapecov = [Variable("x₁", x₁)])
        #
        # fm = Extremes.fit(model)
        #
        #varM = Extremes.parametervar(fm)
        #var = sqrt.([varM[i,i] for i in 1:length(θ)]) .* quantile(Normal(), 0.975)
        #
        #@test fm.θ̂ .- var <= θ
        #@test θ <= fm.θ̂ .+ var

    end

    include(joinpath("maximumlikelihood", "maximumlikelihood_gev_test.jl"))
    include(joinpath("maximumlikelihood", "maximumlikelihood_gp_test.jl"))

end
