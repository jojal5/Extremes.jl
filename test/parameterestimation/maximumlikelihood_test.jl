@testset "maximumlikelihood.jl" begin
    @testset "fit(model, initialvalues)" begin
        n = 5000

        μ = 0.0
        σ = 1.0
        ξ = 0.1

        ϕ = log(σ)
        θ = [μ; ϕ; ξ]

        pd = GeneralizedExtremeValue(μ, σ, ξ)
        y = rand(pd, n)

        model = Extremes.BlockMaxima(Variable("y", y))

        # Initial value vector length != nparameter throws
        @test_throws AssertionError Extremes.fit(model, [0.0, 0.0, 0.0, 0.0])

        # Initial value vector invalid throws
        @test_throws AssertionError Extremes.fit(model, [Inf, Inf, Inf])

        # Initial value vector valid and length == nparameter does not throw
        @test_logs Extremes.fit(model, [0.0, 0.0, 0.0])

    end

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

        model = Extremes.BlockMaxima(Variable("y", y))

        @test_logs (:warn,"The maximum likelihood algorithm did not find a solution. Maybe try with different initial values or with another method. The returned values are the initial values.") Extremes.fit(model)

        # stationary GEV fit by ML
        n = 5000

        μ = 0.0
        σ = 1.0
        ξ = 0.1

        ϕ = log(σ)
        θ = [μ; ϕ; ξ]

        pd = GeneralizedExtremeValue(μ, σ, ξ)
        y = rand(pd, n)

        model = Extremes.BlockMaxima(Variable("y", y))

        fm = Extremes.fit(model)

        varM = Extremes.parametervar(fm)
        var = sqrt.([varM[i,i] for i in 1:length(θ)]) .* quantile(Normal(), 0.975)

        @test fm.θ̂ .- var <= θ
        @test θ <= fm.θ̂ .+ var

        # non-stationary GEV fit by ML
        n = 5000

        x₁ = randn(n)
        x₂ = randn(n)
        x₃ = randn(n) / 10

        μ = 1.0 .+ x₁
        ϕ = -0.5 .+ x₂
        ξ = x₃

        ϕ = log(σ)
        θ = [1.0; 1.0; -0.5; 1.0; 0.0; 1.0]

        pd = GeneralizedExtremeValue.(μ, σ, ξ)
        y = rand.(pd)

        model = Extremes.BlockMaxima(Variable("y", y), locationcov = [Variable("x₁", x₁)], logscalecov = [Variable("x₂", x₂)], shapecov = [Variable("x₃", x₃)])

        fm = Extremes.fit(model)

        varM = Extremes.parametervar(fm)
        var = sqrt.([varM[i,i] for i in 1:length(θ)]) .* quantile(Normal(), 0.975)

        @test fm.θ̂ .- var <= θ
        @test θ <= fm.θ̂ .+ var

        # stationary GP fit by ML
        n = 5000

        σ = 1.0
        ξ = 0.1

        ϕ = log(σ)
        θ = [ϕ; ξ]

        pd = GeneralizedPareto(σ, ξ)
        y = rand(pd, n)

        model = Extremes.ThresholdExceedance(Variable("y", y))

        fm = Extremes.fit(model)

        varM = Extremes.parametervar(fm)
        var = sqrt.([varM[i,i] for i in 1:length(θ)]) .* quantile(Normal(), 0.975)

        @test fm.θ̂ .- var <= θ
        @test θ <= fm.θ̂ .+ var

        # non-stationary GP fit by ML
        n = 5000

        x₁ = randn(n) / 3
        x₂ = randn(n) / 10

        ϕ = -.5 .+ x₁
        ξ = x₂

        σ = exp.(ϕ)
        θ = [-.5; 1.0; 0.0; 1.0]

        pd = GeneralizedPareto.(σ, ξ)
        y = rand.(pd)

        model = Extremes.ThresholdExceedance(Variable("y", y), logscalecov = [Variable("x₁", x₁)], shapecov = [Variable("x₂", x₂)])

        fm = Extremes.fit(model)

        varM = Extremes.parametervar(fm)
        var = sqrt.([varM[i,i] for i in 1:length(θ)]) .* quantile(Normal(), 0.975)

        @test fm.θ̂ .- var <= θ
        @test θ <= fm.θ̂ .+ var

    end

    include(joinpath("maximumlikelihood", "maximumlikelihood_gev_test.jl"))
    include(joinpath("maximumlikelihood", "maximumlikelihood_gp_test.jl"))

end
