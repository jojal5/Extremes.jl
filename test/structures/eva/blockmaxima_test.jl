@testset "blockmaxima.jl" begin
    n = 10000

    pd = GeneralizedExtremeValue(0.0, 1.0, 0.1)
    y = rand(pd, n)

    x = rand(n)

    ev = [ExplanatoryVariable("x", x)]

    smodel = Extremes.BlockMaxima(y)
    nsmodel = Extremes.BlockMaxima(y, locationcov = ev, logscalecov = ev, shapecov = ev)

    @testset "BlockMaxima(data; locationcov, logscalecov, shapecov)" begin
        ev10 = Extremes.ExplanatoryVariable("t", collect(1:n+10))

        # Build with locationcov length != y length
        @test_throws AssertionError Extremes.BlockMaxima(y, locationcov = [ev10])

        # Build with logscalecov length != y length
        @test_throws AssertionError Extremes.BlockMaxima(y, logscalecov = [ev10])

        # Build with shapecov length != y length
        @test_throws AssertionError Extremes.BlockMaxima(y, shapecov = [ev10])

        # Build with all optional parameters set
        @test nsmodel.data == y
        @test nsmodel.location.covariate == ev
        @test nsmodel.logscale.covariate == ev
        @test nsmodel.shape.covariate == ev

    end

    @testset "paramindex(model)" begin
        # model with stationary and non-stationary parameters
        model = Extremes.BlockMaxima(y, locationcov = ev)

        paramin = Extremes.paramindex(model)

        @test length(paramin) == 3
        @test paramin[:μ] == [1, 2]
        @test paramin[:ϕ] == [3]
        @test paramin[:ξ] == [4]

    end

    @testset "getcovariatenumber(model)" begin
        # stationary
        @test Extremes.getcovariatenumber(smodel) == 0

        # non-stationary
        @test Extremes.getcovariatenumber(nsmodel) == 3

    end

    @testset "nparameter(model)" begin
        # stationary
        @test Extremes.nparameter(smodel) == 3

        # non-stationary
        @test Extremes.nparameter(nsmodel) == 6

    end

    @testset "getdistribution(model, θ)" begin
        # length(θ) != nparameter(model) throws
        @test_throws AssertionError Extremes.getdistribution(smodel, [1.0])

        # stationary
        n = 100

        μ = 0.0
        σ = 1.0
        ξ = 0.1
        ϕ = log(σ)

        θ = [μ; ϕ; ξ]

        pd = GeneralizedExtremeValue(μ, σ, ξ)
        y = rand(pd, n)

        model = BlockMaxima(y)

        fd = Extremes.getdistribution(model, θ)[]

        @test fd == pd

        # non-stationary
        n = 10000

        x₁ = randn(n)
        x₂ = randn(n)/3
        x₃ = randn(n)/10

        θ = [5.0 ; 1.0 ; -.5 ; 1.0 ; 0.0 ; 1.0]

        μ = θ[1] .+ θ[2] * x₁
        ϕ = θ[3] .+ θ[4] * x₂
        ξ = θ[5] .+ θ[6] * x₃

        pd = GeneralizedExtremeValue.(μ, exp.(ϕ), ξ)

        y = rand.(pd)

        model = BlockMaxima(y, locationcov = [ExplanatoryVariable("x₁", x₁)], logscalecov = [ExplanatoryVariable("x₂", x₂)], shapecov = [ExplanatoryVariable("x₃", x₃)])

        fd = Extremes.getdistribution(model, θ)

        @test pd == fd

    end

    @testset "getinitialvalue(::Type{GeneralizedExtremeValue},y)" begin
        # TODO : Test with valid_initialvalues (J)

        # TODO : Test with !valid_initialvalues (J)

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
