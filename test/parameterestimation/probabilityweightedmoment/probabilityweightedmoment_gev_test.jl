@testset "probabilityweightedmoment_gev.jl" begin
    n = 5000
    θ = [0.0;0.0;.2]

    pd = GeneralizedExtremeValue(θ[1], exp(θ[2]), θ[3])
    y = rand(pd, n)

    @testset "gevfitpwm(y)" begin
        # stationary model building
        fm = Extremes.gevfitpwm(y)

        varM = Extremes.parametervar(fm)
        var = sqrt.([varM[i,i] for i in 1:length(θ)]) .* quantile(Normal(), 0.975)

        @test fm.θ̂ .- var <= θ
        @test θ <= fm.θ̂ .+ var

    end

    @testset "gevfitpwm(model)" begin
        # non-stationary warn
        model = Extremes.BlockMaxima(y, locationcov = [ExplanatoryVariable("t", collect(1:n))])

        @test_logs (:warn, "covariates cannot be included in the model when estimating the
            paramters by the probability weighted moment parameter estimation.
            The estimates for the stationary model is returned.") Extremes.gevfitpwm(model)

        # stationary GEV fit by pwm
        model = Extremes.BlockMaxima(y)

        fm = Extremes.gevfitpwm(model)

        varM = Extremes.parametervar(fm)
        var = sqrt.([varM[i,i] for i in 1:length(θ)]) .* quantile(Normal(), 0.975)

        @test fm.θ̂ .- var <= θ
        @test θ <= fm.θ̂ .+ var

    end
end
