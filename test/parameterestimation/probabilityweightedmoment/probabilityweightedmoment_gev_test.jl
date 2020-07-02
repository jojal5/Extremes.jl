@testset "probabilityweightedmoment_gev.jl" begin
    n = 5000
    θ = [0.0;0.0;.2]

    pd = GeneralizedExtremeValue(θ[1], exp(θ[2]), θ[3])
    y = rand(pd, n)

    @testset "gevfitpwm(y)" begin
        # stationary model building
        fm = Extremes.gevfitpwm(y)

        cinterval = cint(fm)

        @test [x[1] for x in cinterval] <= θ
        @test θ <= [x[2] for x in cinterval]

    end

    @testset "gevfitpwm(df, datacol)" begin
        # stationary model building
        df = DataFrame(y = y)
        fm = Extremes.gevfitpwm(df, :y)

        cinterval = cint(fm)

        @test [x[1] for x in cinterval] <= θ
        @test θ <= [x[2] for x in cinterval]

    end

    @testset "gevfitpwm(model)" begin
        # non-stationary warn
        model = Extremes.BlockMaxima(Variable("y", y), locationcov = [Variable("t", collect(1:n))])

        @test_logs (:warn, "covariates cannot be included in the model when estimating the
            paramters by the probability weighted moment parameter estimation.
            The estimates for the stationary model is returned.") Extremes.gevfitpwm(model)

        # stationary GEV fit by pwm
        model = Extremes.BlockMaxima(Variable("y", y))

        fm = Extremes.gevfitpwm(model)

        cinterval = cint(fm)

        @test [x[1] for x in cinterval] <= θ
        @test θ <= [x[2] for x in cinterval]

    end
end
