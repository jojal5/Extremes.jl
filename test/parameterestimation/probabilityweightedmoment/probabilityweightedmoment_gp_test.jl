@testset "probabilityweightedmoment_gp.jl" begin
    n = 10000
    θ = [0.0 ; .2]

    pd = GeneralizedPareto(exp(θ[1]), θ[2])
    y = rand(pd, n)

    @testset "gpfitpwm(y)" begin
        # stationary model building
        fm = Extremes.gpfitpwm(y)

        cinterval = cint(fm)

        @test [x[1] for x in cinterval] <= θ
        @test θ <= [x[2] for x in cinterval]

    end

    @testset "gpfitpwm(df, datacol)" begin
        # stationary model building
        df = DataFrame(y = y)
        fm = Extremes.gpfitpwm(df, :y)

        cinterval = cint(fm)

        @test [x[1] for x in cinterval] <= θ
        @test θ <= [x[2] for x in cinterval]

    end

    @testset "gpfitpwm(model)" begin
        # non-stationary warn
        model = Extremes.ThresholdExceedance(Variable("y", y), logscalecov = [Variable("t", collect(1:n))])

        @test_logs (:warn, "covariates cannot be included in the model when estimating the
            paramters by the probability weighted moment parameter estimation.
            The estimates for the stationary model is returned.") Extremes.gpfitpwm(model)

        # stationary GP fit by pwm
        model = Extremes.ThresholdExceedance(Variable("y", y))

        fm = Extremes.gpfitpwm(model)

        cinterval = cint(fm)

        @test [x[1] for x in cinterval] <= θ
        @test θ <= [x[2] for x in cinterval]

    end
end
