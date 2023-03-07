@testset "blockmaxima{GeneralizedExtremeValue}.jl" begin
    n = 2

    μ = 0.0
    σ = 1.0
    ξ = .5

    y = [0., .5]

    x₁ = [0., 1.]
    x₂ = [log(.5), log(1)]

    θ  = [0., 1., 0., 1., 0., .1]

    smodel = Extremes.BlockMaxima{GeneralizedExtremeValue}(Variable("y", y))
    nsmodel = Extremes.BlockMaxima{GeneralizedExtremeValue}(Variable("y", y), locationcov = [Variable("x₁", x₁)], logscalecov = [Variable("x₂", x₂)], shapecov=[Variable("x₁", x₁)])

    @testset "BlockMaxima{GeneralizedExtremeValue}(data; locationcov, logscalecov, shapecov)" begin

        # Build with locationcov length != y length
        @test_throws AssertionError Extremes.BlockMaxima{GeneralizedExtremeValue}(Variable("y", y), locationcov = [Variable("x₁", [1.])])

        # Build with logscalecov length != y length
        @test_throws AssertionError Extremes.BlockMaxima{GeneralizedExtremeValue}(Variable("y", y), logscalecov = [Variable("x₁", [1.])])

        # Build with shapecov length != y length
        @test_throws AssertionError Extremes.BlockMaxima{GeneralizedExtremeValue}(Variable("y", y), shapecov = [Variable("x₁", [1.])])

        # Build with all optional parameters set
        @test nsmodel.data.value == y
        @test nsmodel.location.covariate[1].value == x₁
        @test nsmodel.logscale.covariate[1].value == x₂
        @test nsmodel.shape.covariate[1].value == x₁

    end

    @testset "paramindex(model)" begin

        paramin = Extremes.paramindex(nsmodel)

        @test length(paramin) == 3
        @test paramin[:μ] == [1, 2]
        @test paramin[:ϕ] == [3, 4]
        @test paramin[:ξ] == [5, 6]

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

        @test Extremes.getdistribution(smodel, [μ, log(σ), ξ])[] == GeneralizedExtremeValue(μ, σ, ξ)

        # non-stationary
        
        @test all(Extremes.getdistribution(nsmodel, θ) .== GeneralizedExtremeValue.(θ[1] .+ θ[2]*x₁, exp.(θ[3] .+ θ[4]*x₂), θ[5] .+ θ[6]*x₁))

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
        model = BlockMaxima{GeneralizedExtremeValue}(Variable("y", y))
        ini = Extremes.getinitialvalue(model)
        @test ini[1] ≈ .586 atol = .001
        @test ini[2] ≈ .164 atol = .001
        @test ini[3] ≈ -.285 atol = .001

        # Test with invalid pwm GEV estimates
        y = [0.0 , 1.0, 1.0, 1.0, 3.0,-5.0]
        model = BlockMaxima{GeneralizedExtremeValue}(Variable("y", y))
        ini = Extremes.getinitialvalue(model)
        @test ini[1] ≈ -1.027 atol = .001
        @test ini[2] ≈ 0.726 atol = .001
        @test ini[3] ≈ 0 atol = .001

    end

    @testset "showEVA(io, obj; prefix)" begin
        # print does not throw
        buffer = IOBuffer()
        @test_logs Extremes.showEVA(buffer, smodel, prefix = "\t")
    end

end

