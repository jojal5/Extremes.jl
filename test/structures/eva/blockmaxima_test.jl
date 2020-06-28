@testset "blockmaxima.jl" begin
    n = 10000

    pd = GeneralizedExtremeValue(0.0, 1.0, 0.1)
    y = rand(pd, n)

    x = rand(n)

    ev = [Variable("x", x)]

    smodel = Extremes.BlockMaxima(Variable("y", y))
    nsmodel = Extremes.BlockMaxima(Variable("y", y), locationcov = ev, logscalecov = ev, shapecov = ev)

    @testset "BlockMaxima(data; locationcov, logscalecov, shapecov)" begin
        ev10 = Extremes.Variable("t", collect(1:n+10))

        # Build with locationcov length != y length
        @test_throws AssertionError Extremes.BlockMaxima(Variable("y", y), locationcov = [ev10])

        # Build with logscalecov length != y length
        @test_throws AssertionError Extremes.BlockMaxima(Variable("y", y), logscalecov = [ev10])

        # Build with shapecov length != y length
        @test_throws AssertionError Extremes.BlockMaxima(Variable("y", y), shapecov = [ev10])

        # Build with all optional parameters set
        @test nsmodel.data.value == y
        @test nsmodel.location.covariate == ev
        @test nsmodel.logscale.covariate == ev
        @test nsmodel.shape.covariate == ev

    end

    @testset "paramindex(model)" begin
        # model with stationary and non-stationary parameters
        model = Extremes.BlockMaxima(Variable("y", y), locationcov = ev)

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

        model = BlockMaxima(Variable("y", y))

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

        model = BlockMaxima(Variable("y", y), locationcov = [Variable("x₁", x₁)], logscalecov = [Variable("x₂", x₂)], shapecov = [Variable("x₃", x₃)])

        fd = Extremes.getdistribution(model, θ)

        @test pd == fd

    end

    @testset "getinitialvalue(::Type{GeneralizedExtremeValue},y)" begin
        # Test with valid pwm GEV estimates
        y = [0.0, 1.0, 2.0]
        ini = Extremes.getinitialvalue(GeneralizedExtremeValue,y)
        @test !isapprox(ini[3],0)
        pd = GeneralizedExtremeValue(ini[1], exp(ini[2]), ini[3])
        @test all(insupport.(pd,y))

        # Test with invalid pwm GEV estimates
        y = [0.0 , 1.0, 1.0, 1.0, 3.0,-5.0]
        ini = Extremes.getinitialvalue(GeneralizedExtremeValue,y)
        @test ini[3] ≈ 0
        pd = GeneralizedExtremeValue(ini[1], exp(ini[2]), ini[3])
        @test all(insupport.(pd,y))

    end

    @testset "getinitialvalue(model)" begin
        # Test with valid pwm GEV estimates
        y = [0.0, 1.0, 2.0]
        model = BlockMaxima(Variable("y", y))
        ini = Extremes.getinitialvalue(model)
        @test ini[1] ≈ .586 atol = .001
        @test ini[2] ≈ .164 atol = .001
        @test ini[3] ≈ -.285 atol = .001

        # Test with invalid pwm GEV estimates
        y = [0.0 , 1.0, 1.0, 1.0, 3.0,-5.0]
        model = BlockMaxima(Variable("y", y))
        ini = Extremes.getinitialvalue(model)
        @test ini[1] ≈ -1.027 atol = .001
        @test ini[2] ≈ -.319 atol = .001
        @test ini[3] ≈ 0 atol = .001

    end

    @testset "showEVA(io, obj; prefix)" begin
        # print does not throw
        buffer = IOBuffer()
        @test_logs Extremes.showEVA(buffer, smodel, prefix = "\t")
    end

end
