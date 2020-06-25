@testset "probabilityweightedmoment_gumbel.jl" begin
    n = 5000
    θ = [0.0 ; 0.0]

    pd = Gumbel(θ[1], exp(θ[2]))
    y = rand(pd, n)

    @testset "gumbelfitpwm(y)" begin
        # stationary model building
        fm = Extremes.gumbelfitpwm(y)

        varM = Extremes.parametervar(fm)
        var = sqrt.([varM[i,i] for i in 1:length(θ)]) .* quantile(Normal(), 0.975)

        @test fm.θ̂[1:2] .- var[1:2] <= θ
        @test θ <= fm.θ̂[1:2] .+ var[1:2]

    end

    @testset "gumbelfitpwm(y)" begin
        # stationary model building
        df = DataFrame(y = y)
        fm = Extremes.gumbelfitpwm(df, :y)

        varM = Extremes.parametervar(fm)
        var = sqrt.([varM[i,i] for i in 1:length(θ)]) .* quantile(Normal(), 0.975)

        @test fm.θ̂[1:2] .- var[1:2] <= θ
        @test θ <= fm.θ̂[1:2] .+ var[1:2]

    end

    @testset "gumbelfitpwm(model)" begin
        # non-stationary warn
        model = Extremes.BlockMaxima(Variable("y", y), locationcov = [Variable("t", collect(1:n))], dist = Gumbel)

        @test_logs (:warn, "covariates cannot be included in the model when estimating the
            paramters by the probability weighted moment parameter estimation.
            The estimates for the stationary model is returned.") Extremes.gumbelfitpwm(model)

        # stationary Gumbel fit by pwm
        model = Extremes.BlockMaxima(Variable("y", y), dist = Gumbel)

        fm = Extremes.gumbelfitpwm(model)

        varM = Extremes.parametervar(fm)
        var = sqrt.([varM[i,i] for i in 1:length(θ)]) .* quantile(Normal(), 0.975)

        @test fm.θ̂[1:2] .- var[1:2] <= θ
        @test θ <= fm.θ̂[1:2] .+ var[1:2]

    end
end
