@testset "eva.jl" begin
    @testset "computeparamfunction(covariates)" begin
        # function returned for empty covariates should not modify θ
        θ = [1.0]
        fsame = Extremes.computeparamfunction(Variable[])
        θtransformed = fsame(θ)

        @test θ == θtransformed

        # function returned for not empty covariates should modify θ
        θ = [1.0, 1.0]
        ev = Variable("t", collect(1:100))
        ftrans = Extremes.computeparamfunction([ev])
        θtransformed = ftrans(θ)

        @test θ != θtransformed

    end

    x₁ = Variable("x₁", [1])
    x₂ = Variable("x₂", [2])
    x₃ = Variable("x₂", [3])

    μ = 1 + x₂.value[] + x₃.value[]
    ϕ = -.5 + x₁.value[]
    ξ = .1

    y = [6]

    pd = GeneralizedExtremeValue(μ, exp(ϕ), ξ)

    model = BlockMaxima{GeneralizedExtremeValue}(Variable("y", y), locationcov=[x₂; x₃], logscalecov = [x₁])

    @testset "loglike(model, θ)" begin


        # Test BlockMaxima with known values
        pd = GeneralizedExtremeValue(μ, exp(ϕ), ξ)
        model = BlockMaxima{GeneralizedExtremeValue}(Variable("y", y), locationcov=[x₂; x₃], logscalecov = [x₁])
        @test Extremes.loglike(model, [1; 1; 1; -.5; 1; .1]) ≈ logpdf(pd,y[])

        # Test ThresholdExceedance with known values
        pd = GeneralizedPareto(exp(ϕ), ξ)
        model = ThresholdExceedance(Variable("y", y), logscalecov = [x₁])
        @test Extremes.loglike(model, [-.5; 1; .1]) ≈ logpdf(pd,y[])

    end

    @testset "quantile(model, θ, p)" begin
        #  p outside of [0, 1] throws
        @test_throws AssertionError quantile(model, [-.5; 1; .1], -1)

        # Test BlockMaxima with known values
        pd = GeneralizedExtremeValue(μ, exp(ϕ), ξ)
        model = BlockMaxima{GeneralizedExtremeValue}(Variable("y", y), locationcov=[x₂; x₃], logscalecov = [x₁])
        @test Extremes.quantile(model, [1; 1; 1; -.5; 1; .1],.99)[] ≈ quantile(pd,.99)

        # Test ThresholdExceedance with known values
        pd = GeneralizedPareto(exp(ϕ), ξ)
        model = ThresholdExceedance(Variable("y", y), logscalecov = [x₁])
        @test Extremes.quantile(model, [-.5; 1; .1], .99)[] ≈ quantile(pd,.99)

    end

    @testset "validatestationarity(model)" begin
        n = 1000
        y = collect(1:n)
        ev = [Variable("x", rand(n))]
        smodel = nothing

        # non-stationary BlockMaxima model
        model = Extremes.BlockMaxima{GeneralizedExtremeValue}(Variable("y", y), locationcov = ev, logscalecov = ev, shapecov = ev)
        @test_logs (:warn, "covariates cannot be included in the model when estimating the
            paramters by the probability weighted moment parameter estimation.
            The estimates for the stationary model is returned.") smodel = Extremes.validatestationarity(model)

        @test length(smodel.location.covariate) == 0
        @test length(smodel.logscale.covariate) == 0
        @test length(smodel.shape.covariate) == 0

        # non-stationary ThresholdExceedance model
        model = Extremes.ThresholdExceedance(Variable("y", y), logscalecov = ev, shapecov = ev)
        @test_logs (:warn, "covariates cannot be included in the model when estimating the
            paramters by the probability weighted moment parameter estimation.
            The estimates for the stationary model is returned.") smodel = Extremes.validatestationarity(model)

        @test length(smodel.logscale.covariate) == 0
        @test length(smodel.shape.covariate) == 0

        # stationary BlockMaxima model
        model = Extremes.BlockMaxima{GeneralizedExtremeValue}(Variable("y", y))
        @test_logs smodel = Extremes.validatestationarity(model)

        @test length(smodel.location.covariate) == 0
        @test length(smodel.logscale.covariate) == 0
        @test length(smodel.shape.covariate) == 0

        # stationary ThresholdExceedance model
        model = Extremes.ThresholdExceedance(Variable("y", y))
        @test_logs smodel = Extremes.validatestationarity(model)

        @test length(smodel.logscale.covariate) == 0
        @test length(smodel.shape.covariate) == 0

    end

    @testset "Base.show(io, obj)" begin
        # print BlockMaxima does not throw
        model = Extremes.BlockMaxima{GeneralizedExtremeValue}(Variable("y", collect(1:100)))
        buffer = IOBuffer()
        @test_logs Base.show(buffer, model)

        # print ThresholdExceedance does not throw
        model = Extremes.ThresholdExceedance(Variable("y", collect(1:100)))
        buffer = IOBuffer()
        @test_logs Base.show(buffer, model)

    end

    @testset "showparamfun(name, param)" begin
        f(β) = identity(β)
        # stationary param
        param = Extremes.paramfun(Variable[], f)

        @test Extremes.showparamfun("μ", param) == "μ ~ 1"

        # non-stationary param
        n = 100
        param = Extremes.paramfun([Variable("t", collect(1:n)), Variable("x", rand(n))], f)

        @test Extremes.showparamfun("μ", param) == "μ ~ 1 + t + x"

    end

    @testset "validatelength(length, explvariables)" begin
        n = 100
        evn = Variable("tn", collect(1:n))
        evnx = Variable("xn", rand(n))
        evn10 = Variable("tn10", collect(1:n+10))
        # empty explvariables shold not throw
        @test_logs Extremes.validatelength(n, Variable[])

        # explvariables length all == length should not throw
        @test_logs Extremes.validatelength(n, [evn, evnx])

        # first explanatory variable length != length throws
        @test_throws AssertionError Extremes.validatelength(n, [evn10, evn])

        # second explanatory variable length != length throws
        @test_throws AssertionError Extremes.validatelength(n, [evn, evn10])

    end

    include(joinpath("eva", "blockmaxima_test.jl"))
    include(joinpath("eva", "thresholdexceedance_test.jl"))

end
