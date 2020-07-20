@testset "bayesianeva.jl" begin

    n = 1000
    nobservation = 1000
    nobsperblock = 1

    threshold = 10.0

    x = Variable("x",randn(n))

    θ = [-.5; 1; 0.1]

    ϕ = θ[1] .+ θ[2]*x.value
    σ = exp.(ϕ)
    ξ = θ[3]

    pd = GeneralizedPareto.(σ, ξ)

    y = rand.(pd)

    model = ThresholdExceedance(Variable("y", y), logscalecov = [x])

    fm = Extremes.BayesianEVA(model, Mamba.Chains(collect(θ')))

    rl = returnlevel(fm, 0, nobservation, nobsperblock, 100)

    ci = cint(rl)

    @testset "getdistribution(fittedmodel)" begin
        @test all(vec(Extremes.getdistribution(fm)) .== pd)
    end


    @testset "quantile(fm, p)" begin
        # p not in [0, 1] throws
        @test_throws AssertionError Extremes.quantile(fm, -1)

        # Test with known values
        @test vec(quantile(fm, .95)) ≈ quantile.(pd, .95)
    end

    @testset "returnlevel(fm, returnPeriod)" begin
        # returnPeriod < 0 throws
        @test_throws AssertionError returnlevel(fm, 0, nobservation, nobsperblock, -100)

        # Test with known values
        p = 1-nobservation/(100 * nobsperblock * n)
        @test vec(rl.value) ≈ quantile.(pd, p)
    end


    @testset "cint(fm, returnPeriod, confidencelevel)" begin
        # confidencelevel not in [0, 1]
        @test_throws AssertionError cint(returnlevel(fm, 0, nobservation, nobsperblock, 100), -1)

        # Test with known values
        @test ci[1][1] ≈ rl.value[1,1]
        @test ci[5][1] ≈ rl.value[1,5]

    end



    @testset "showfittedEVA(io, obj, prefix)" begin
        # print does not throw
        buffer = IOBuffer()
        @test_logs Extremes.showfittedEVA(buffer, fm, prefix = "\t")
    end

    @testset "showChain(io, chain, prefix)" begin
        # print does not throw
        buffer = IOBuffer()
        @test_logs Extremes.showChain(buffer, fm.sim)
    end


    @testset "findposteriormode(fm::BayesianEVA)" begin

        x = Variable("x", randn(10))
        μ = 10 .+ x.value
        σ = 1.0
        ξ = .1
        pd = GeneralizedExtremeValue.(μ, σ, ξ)
        y = rand.(pd)
        fm = Extremes.BayesianEVA(Extremes.BlockMaxima(Variable("y", y), locationcov=[x]),
            Mamba.Chains([10.0 1.0 0.0 .1; -10.0 1.0 0.0 .1; 20.0 1.0 0.0 .1]))

        θ̂ = Extremes.findposteriormode(fm)
        @test θ̂ ≈ [10.0; 1.0; 0.0; .1]

    end

end
