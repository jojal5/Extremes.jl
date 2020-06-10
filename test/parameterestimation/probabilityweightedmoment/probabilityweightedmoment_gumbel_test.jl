@testset "probabilityweightedmoment_gumbel.jl" begin
    @testset "gumbelfitpwm(y)" begin

    end

    @testset "gumbelfitpwm(model)" begin
        # stationary Gumbel fit by pwm
        n = 10000
        θ = [0.0 ; 1.0]

        pd = Gumbel(θ...)
        y = rand(pd, n)

        model = Extremes.BlockMaxima(y)

        fm = Extremes.gumbelfitpwm(model)

        @test fm.θ̂[1:2] ≈ θ rtol = .05

    end
end
