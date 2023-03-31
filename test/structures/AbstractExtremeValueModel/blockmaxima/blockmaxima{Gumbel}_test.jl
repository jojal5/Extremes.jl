@testset "blockmaxima{Gumbel}.jl" begin
    n = 2

    μ = 0.0
    σ = 1.0

    y = [0., .5]

    x₁ = [0., 1.]
    x₂ = [log(.5), log(1)]

    θ  = [0., 1., 0., 1.]

    smodel = Extremes.BlockMaxima{Gumbel}(Variable("y", y))
    nsmodel = Extremes.BlockMaxima{Gumbel}(Variable("y", y), locationcov = [Variable("x₁", x₁)], logscalecov = [Variable("x₂", x₂)])

    @testset "BlockMaxima{Gumbel}(data; locationcov, logscalecov)" begin

        # Build with locationcov length != y length
        @test_throws AssertionError Extremes.BlockMaxima{Gumbel}(Variable("y", y), locationcov = [Variable("x₁", [1.])])

        # Build with logscalecov length != y length
        @test_throws AssertionError Extremes.BlockMaxima{Gumbel}(Variable("y", y), logscalecov = [Variable("x₂", [1.])])


        # Build with all optional parameters set
        @test nsmodel.data.value == y
        @test nsmodel.location.covariate[1].value == x₁
        @test nsmodel.logscale.covariate[1].value == x₂

    end

    @testset "paramindex(model)" begin
        # model with stationary and non-stationary parameters
        # model = Extremes.BlockMaxima{Gumbel}(Variable("y", y), locationcov = ev)

        paramin = Extremes.paramindex(nsmodel)

        @test length(paramin) == 2
        @test paramin[:μ] == [1, 2]
        @test paramin[:ϕ] == [3, 4]

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

        fd = Extremes.getdistribution(smodel, [μ, log(σ)])[]

        @test fd == Gumbel(μ, σ)

        # non-stationary

        pd = Gumbel.(θ[1] .+ θ[2]*x₁, exp.(θ[3] .+ θ[4]*x₂))

        model = BlockMaxima{Gumbel}(Variable("y", y), locationcov = [Variable("x₁", x₁)], logscalecov = [Variable("x₂", x₂)])

        fd = Extremes.getdistribution(model, θ)

        @test pd == fd

    end

    
    @testset "getinitialvalue(model)" begin
        
        ini = Extremes.getinitialvalue(smodel)
        @test ini[1] ≈ .042 atol = .001
        @test ini[2] ≈ .-1.020 atol = .001

        ini = Extremes.getinitialvalue(nsmodel)
        @test ini[1] ≈ .042 atol = .001
        @test ini[2] ≈ 0. atol = .001
        @test ini[3] ≈ .-1.020 atol = .001
        @test ini[4] ≈ 0. atol = .001

    end

    @testset "showAbstractExtremeValueModel(io, obj; prefix)" begin
        # print does not throw
        buffer = IOBuffer()
        @test_logs Extremes.showAbstractExtremeValueModel(buffer, smodel, prefix = "\t")
    end

end

