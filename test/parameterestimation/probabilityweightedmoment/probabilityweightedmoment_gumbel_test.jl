@testset "probabilityweightedmoment_gumbel.jl" begin
    n = 5000
    θ = [0.0 ; 0.0]

    pd = Gumbel(θ[1], exp(θ[2]))
    y = rand(pd, n)

    @testset "gumbelfitpwm(y)" begin
        # stationary model building
        fm = Extremes.gumbelfitpwm(y)

        cinterval = cint(fm)

        @test [x[1] for x in cinterval] <= θ
        @test θ <= [x[2] for x in cinterval]

    end

    @testset "gumbelfitpwm(df, datacol)" begin
        # stationary model building
        df = DataFrame(y = y)
        fm = Extremes.gumbelfitpwm(df, :y)

        cinterval = cint(fm)

        @test [x[1] for x in cinterval] <= θ
        @test θ <= [x[2] for x in cinterval]

    end

    @testset "gumbelfitpwm(model)" begin
        # non-stationary warn
        model = Extremes.BlockMaxima(Variable("y", y), locationcov = [Variable("t", collect(1:n))])

        @test_logs (:warn, "covariates cannot be included in the model when estimating the
            paramters by the probability weighted moment parameter estimation.
            The estimates for the stationary model is returned.") Extremes.gumbelfitpwm(model)

        # stationary Gumbel fit by pwm
        model = Extremes.BlockMaxima(Variable("y", y))

        fm = Extremes.gumbelfitpwm(model)

        cinterval = cint(fm)

        @test [x[1] for x in cinterval] <= θ
        @test θ <= [x[2] for x in cinterval]

    end
end
