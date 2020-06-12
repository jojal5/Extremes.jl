@testset "probabilityweightedmoment_gev.jl" begin
    @testset "gevfitpwm(y)" begin
        # TODO : add test for model building

    end

    @testset "gevfitpwm(model)" begin
        # TODO : Add non-stationary warn test

        # stationary GEV fit by pwm
        n = 10000
        θ = [0.0;0.0;.2]

        pd = GeneralizedExtremeValue(θ[1], exp(θ[2]), θ[3])
        y = rand(pd, n)

        model = Extremes.BlockMaxima(y)

        fm = Extremes.gevfitpwm(model)

        @test fm.θ̂ ≈ θ rtol = .05

    end
end
