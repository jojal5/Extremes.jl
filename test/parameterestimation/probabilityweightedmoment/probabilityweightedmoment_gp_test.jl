@testset "probabilityweightedmoment_gp.jl" begin
    n = 10000
    θ = [0.0 ; .2]

    pd = GeneralizedPareto(exp(θ[1]), θ[2])
    y = rand(pd, n)

    @testset "gpfitpwm(y)" begin
        # stationary model building
        fm = Extremes.gpfitpwm(y)

        @test fm.θ̂ ≈ θ rtol = .05

    end

    @testset "gpfitpwm(model)" begin
        # non-stationary warn
        model = Extremes.ThresholdExceedance(y, logscalecov = [ExplanatoryVariable("t", collect(1:n))])

        @test_logs (:warn, "covariates cannot be included in the model when estimating the
            paramters by the probability weighted moment parameter estimation.
            The estimates for the stationary model is returned.") Extremes.gpfitpwm(model)

        # stationary GP fit by pwm
        model = Extremes.ThresholdExceedance(y)

        fm = Extremes.gpfitpwm(model)

        @test fm.θ̂ ≈ θ rtol = .05

    end
end
