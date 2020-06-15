@testset "thresholdexceedance.jl" begin
    n = 10000

    pd = GeneralizedPareto(1.0, 0.1)
    y = rand(pd, n)

    x = rand(n)

    ev = [ExplanatoryVariable("x", x)]

    smodel = Extremes.ThresholdExceedance(y)
    nsmodel = Extremes.ThresholdExceedance(y, logscalecov = ev, shapecov = ev)

    @testset "ThresholdExceedance(exceedances; logscalecov, shapecov)" begin
        ev10 = Extremes.ExplanatoryVariable("t", collect(1:n+10))

        # Build with logscalecov length != y length
        @test_throws AssertionError Extremes.ThresholdExceedance(y, logscalecov = [ev10])

        # Build with shapecov length != y length
        @test_throws AssertionError Extremes.ThresholdExceedance(y, shapecov = [ev10])

        # Build with all optional parameters set
        @test nsmodel.data == y
        @test nsmodel.logscale.covariate == ev
        @test nsmodel.shape.covariate == ev
    end

    @testset "paramindex(model)" begin
        # model with stationary and non-stationary parameters
        model = Extremes.ThresholdExceedance(y, logscalecov = ev)

        paramin = Extremes.paramindex(model)

        @test length(paramin) == 2
        @test paramin[:ϕ] == [1,2]
        @test paramin[:ξ] == [3]
    end

    @testset "getcovariatenumber(model)" begin
        # stationary
        @test Extremes.getcovariatenumber(smodel) == 0

        # non-stationary
        @test Extremes.getcovariatenumber(nsmodel) == 2
    end

    @testset "nparameter(model)" begin
        # stationary
        @test Extremes.nparameter(smodel) == 2

        # non-stationary
        @test Extremes.nparameter(nsmodel) == 4
    end

    @testset "getdistribution(model, θ)" begin
        # length(θ) != nparameter(model) throws
        @test_throws AssertionError Extremes.getdistribution(smodel, [1.0])

        # stationary
        n = 100

        σ = 1.0
        ξ = .1
        ϕ = log(σ)

        θ = [ϕ ; ξ]

        pd = GeneralizedPareto(σ, ξ)
        y = rand(pd,n)

        model = ThresholdExceedance(y)

        fd = Extremes.getdistribution(model, θ)[]

        @test fd == pd

        # non-stationary
        n = 100

        x₁ = randn(n)/3
        x₂ = randn(n)/3
        x₃ = randn(n)/10

        θ = [-.5 ; 1.0 ; 1.0 ; 0 ; 1.0]

        ϕ = θ[1] .+ θ[2] * x₁ .+ θ[3] * x₂
        ξ = θ[4] .+ θ[5] * x₃

        σ = exp.(ϕ)

        pd = GeneralizedPareto.(σ, ξ)
        y = rand.(pd)

        model = ThresholdExceedance(y, logscalecov = [ExplanatoryVariable("x₁", x₁), ExplanatoryVariable("x₂", x₂)], shapecov = [ExplanatoryVariable("x₃", x₃)])

        fd = Extremes.getdistribution(model, θ)

        @test fd == pd

    end

    @testset "getinitialvalue(::Type{GeneralizedPareto},y)" begin
        # TODO : Test with all(insupport(fd,y)) (J)

        # TODO : Test with !all(insupport(fd,y)) (J)

    end

    @testset "getinitialvalue(model)" begin
        # TODO : Test with known values (J)

    end

    @testset "showEVA(io, obj; prefix)" begin
        # print does not throw
        buffer = IOBuffer()
        @test_logs Extremes.showEVA(buffer, smodel, prefix = "\t")

    end

end
