@testset "probabilityweightedmoment_gp.jl" begin
    n = 10000
    θ = [0.0 ; .2]

    pd = GeneralizedPareto(exp(θ[1]), θ[2])
    y = rand(pd, n)

    @testset "gpfitpwm(y)" begin
        # stationary model building
        fm = Extremes.gpfitpwm(y)

        varM = Extremes.parametervar(fm)
        var = sqrt.([varM[i,i] for i in 1:length(θ)]) .* quantile(Normal(), 0.975)

        @test fm.θ̂ .- var <= θ
        @test θ <= fm.θ̂ .+ var

    end

    @testset "gpfitpwm(model)" begin
        # non-stationary warn
        model = Extremes.ThresholdExceedance(y, logscalecov = [Variable("t", collect(1:n))])

        @test_logs (:warn, "covariates cannot be included in the model when estimating the
            paramters by the probability weighted moment parameter estimation.
            The estimates for the stationary model is returned.") Extremes.gpfitpwm(model)

        # stationary GP fit by pwm
        model = Extremes.ThresholdExceedance(y)

        fm = Extremes.gpfitpwm(model)

        varM = Extremes.parametervar(fm)
        var = sqrt.([varM[i,i] for i in 1:length(θ)]) .* quantile(Normal(), 0.975)

        @test fm.θ̂ .- var <= θ
        @test θ <= fm.θ̂ .+ var

    end
end
