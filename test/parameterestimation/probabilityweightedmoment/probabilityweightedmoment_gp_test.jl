@testset "probabilityweightedmoment_gp.jl" begin
    @testset "gpfitpwm(y)" begin
        # TODO : add test for model building

    end

    @testset "gpfitpwm(model)" begin
        # TODO : Add non-stationary warn test

        # stationary GP fit by pwm
        n = 10000
        θ = [0.0 ; .2]

        pd = GeneralizedPareto(exp(θ[1]), θ[2])
        y = rand(pd, n)

        model = Extremes.ThresholdExceedance(y)

        fm = Extremes.gpfitpwm(model)

        @test fm.θ̂ ≈ θ rtol = .05

    end
end
