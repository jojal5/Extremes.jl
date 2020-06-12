@testset "eva.jl" begin
    @testset "computeparamfunction(covariates)" begin
        # function returned for empty covariates should not modify θ
        θ = [1.0]
        fsame = Extremes.computeparamfunction(ExplanatoryVariable[])
        θtransformed = fsame(θ)

        @test θ == θtransformed

        # function returned for not empty covariates should modify θ
        θ = [1.0, 1.0]
        ev = ExplanatoryVariable("t", collect(1:100))
        ftrans = Extremes.computeparamfunction([ev])
        θtransformed = ftrans(θ)

        @test θ != θtransformed

    end

    @testset "loglike(model, θ)" begin
        # TODO: Test BlockMaxima with known values (J)

        # TODO: Test ThresholdExceedance with known values (J)

    end

    @testset "quantile(model, θ, p)" begin
        #  p outside of [0, 1] throws
        n = 1000
        θ = [0.0, 1.0, 0.1]
        pd = GeneralizedExtremeValue(θ...)
        y = rand(pd, n)
        model = Extremes.BlockMaxima(y)

        @test_throws AssertionError quantile(model, θ, -1)

        # TODO: Test BlockMaxima with known values (J)

        # TODO: Test ThresholdExceedance with known values (J)

    end

    @testset "validatestationarity(model)" begin
        n = 1000
        y = collect(1:n)
        ev = [ExplanatoryVariable("x", rand(n))]
        smodel = nothing

        # non-stationary BlockMaxima model
        model = Extremes.BlockMaxima(y, locationcov = ev, logscalecov = ev, shapecov = ev)
        @test_logs (:warn, "covariates cannot be included in the model when estimating the
            paramters by the probability weighted moment parameter estimation.
            The estimates for the stationary model is returned.") smodel = Extremes.validatestationarity(model)

        @test length(smodel.location.covariate) == 0
        @test length(smodel.logscale.covariate) == 0
        @test length(smodel.shape.covariate) == 0

        # non-stationary ThresholdExceedance model
        model = Extremes.ThresholdExceedance(y, logscalecov = ev, shapecov = ev)
        @test_logs (:warn, "covariates cannot be included in the model when estimating the
            paramters by the probability weighted moment parameter estimation.
            The estimates for the stationary model is returned.") smodel = Extremes.validatestationarity(model)

        @test length(smodel.logscale.covariate) == 0
        @test length(smodel.shape.covariate) == 0

        # stationary BlockMaxima model
        model = Extremes.BlockMaxima(y)
        @test_logs smodel = Extremes.validatestationarity(model)

        @test length(smodel.location.covariate) == 0
        @test length(smodel.logscale.covariate) == 0
        @test length(smodel.shape.covariate) == 0

        # stationary ThresholdExceedance model
        model = Extremes.ThresholdExceedance(y)
        @test_logs smodel = Extremes.validatestationarity(model)

        @test length(smodel.logscale.covariate) == 0
        @test length(smodel.shape.covariate) == 0

    end

    @testset "Base.show(io, obj)" begin
        # print BlockMaxima does not throw
        model = Extremes.BlockMaxima(collect(1:100))
        buffer = IOBuffer()
        @test_logs Base.show(buffer, model)

        # print ThresholdExceedance does not throw
        model = Extremes.ThresholdExceedance(collect(1:100))
        buffer = IOBuffer()
        @test_logs Base.show(buffer, model)

    end

    @testset "showparamfun(name, param)" begin
        f(β) = identity(β)
        # stationary param
        param = Extremes.paramfun(ExplanatoryVariable[], f)

        @test Extremes.showparamfun("μ", param) == "μ ~ 1"

        # non-stationary param
        n = 100
        param = Extremes.paramfun([ExplanatoryVariable("t", collect(1:n)), ExplanatoryVariable("x", rand(n))], f)

        @test Extremes.showparamfun("μ", param) == "μ ~ 1 + t + x"

    end

    include(joinpath("eva", "blockmaxima_test.jl"))
    include(joinpath("eva", "thresholdexceedance_test.jl"))

end
