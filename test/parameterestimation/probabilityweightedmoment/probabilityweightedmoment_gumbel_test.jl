@testset "probabilityweightedmoment_gumbel.jl" begin
    n = 10000
    θ = [0.0 ; 0.0]

    pd = Gumbel(θ[1], exp(θ[2]))
    y = rand(pd, n)

    @testset "gumbelfitpwm(y)" begin
        # stationary model building
        fm = Extremes.gumbelfitpwm(y)

        @test fm.θ̂[1:2] ≈ θ rtol = .05

    end

    @testset "gumbelfitpwm(model)" begin
        # non-stationary warn
        model = Extremes.BlockMaxima(y, locationcov = [ExplanatoryVariable("t", collect(1:n))])

        @test_logs (:warn, "covariates cannot be included in the model when estimating the
            paramters by the probability weighted moment parameter estimation.
            The estimates for the stationary model is returned.") Extremes.gumbelfitpwm(model)

        # stationary Gumbel fit by pwm
        model = Extremes.BlockMaxima(y)

        fm = Extremes.gumbelfitpwm(model)

        @test fm.θ̂[1:2] ≈ θ rtol = .05

    end
end
