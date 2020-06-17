@testset "maximumlikelihood_gev.jl" begin
    n = 10000

    x₁ = randn(n)
    x₂ = randn(n) / 3
    x₃ = randn(n) / 10

    μ = 1.0 .+ x₁
    ϕ = -.05 .+ x₂
    ξ = x₃

    σ = exp.(ϕ)
    θ = [1.0; 1.0; -0.05; 1.0; 0.0; 1.0]

    pd = GeneralizedExtremeValue.(μ, σ, ξ)
    y = rand.(pd)

    @testset "gevfit(y; locationcov, logscalecov, shapecov)" begin
        # model building with non-stationary location, logscale and shape
        fm = Extremes.gevfit(y,
            locationcov = [ExplanatoryVariable("x₁", x₁)],
            logscalecov = [ExplanatoryVariable("x₂", x₂)],
            shapecov = [ExplanatoryVariable("x₃", x₃)])

        varM = Extremes.parametervar(fm)
        var = sqrt.([varM[i,i] for i in 1:length(θ)]) .* quantile(Normal(), 0.975)

        @test fm.θ̂ .- var <= θ
        @test θ <= fm.θ̂ .+ var

    end

    @testset "gevfit(df, datacol; locationcovid, logscalecovid, shapecovid)" begin
        # model building with non-stationary location, logscale and shape
        df = DataFrame(y = y, x1 = x₁, x2 = x₂, x3 = x₃)

        fm = Extremes.gevfit(df, :y, locationcovid = [:x1], logscalecovid = [:x2], shapecovid = [:x3])

        varM = Extremes.parametervar(fm)
        var = sqrt.([varM[i,i] for i in 1:length(θ)]) .* quantile(Normal(), 0.975)

        @test fm.θ̂ .- var <= θ
        @test θ <= fm.θ̂ .+ var

    end

    @testset "gevfit(model)" begin
        # non-stationary location, logscale and shape
        model = Extremes.BlockMaxima(y,
            locationcov = [ExplanatoryVariable("x₁", x₁)],
            logscalecov = [ExplanatoryVariable("x₂", x₂)],
            shapecov = [ExplanatoryVariable("x₃", x₃)])

        fm = Extremes.gevfit(model)

        varM = Extremes.parametervar(fm)
        var = sqrt.([varM[i,i] for i in 1:length(θ)]) .* quantile(Normal(), 0.975)

        @test fm.θ̂ .- var <= θ
        @test θ <= fm.θ̂ .+ var

    end

end
