@testset "maximumlikelihoodAbstractExtremeValueModel.jl" begin

    df = CSV.read("dataset/gev_nonstationary.csv", DataFrame)

    deleteat!(df, 101:nrow(df))

    model = Extremes.BlockMaxima{GeneralizedExtremeValue}(Variable("y", df.y),
        locationcov = [Variable("x₁", df.x₁)],
        logscalecov = [Variable("x₂", df.x₂)],
        shapecov = [Variable("x₃", df.x₃)])

    θ = [1., 1., -.5, 1., 0.001, 0.]

    k = 6

    fm = MaximumLikelihoodAbstractExtremeValueModel(model, θ)

    μ = θ[1] .+ θ[2].*df.x₁
    ϕ = θ[3] .+ θ[4].*df.x₂
    ξ = θ[5] .+ θ[6].*df.x₃

    pd = GeneralizedExtremeValue.(μ, exp.(ϕ), ξ)

    @testset "aic" begin

        @test aic(fm) ≈ 2*k - 2*sum(logpdf.(pd, df.y))
        
    end

    @testset "bic" begin

        n = length(pd)
        @test bic(fm) ≈ k*log(n) - 2*sum(logpdf.(pd, df.y))
        
    end

    @testset "getdistribution(fittedmodel)" begin
        @test all(Extremes.getdistribution(fm) .== pd)
    end

    @testset "loglike(fd)" begin
        # Test with known values
        @test Extremes.loglike(fm) ≈ sum(logpdf.(pd, df.y))
    end

    @testset "quantile(fm, p)" begin
        # p not in [0, 1] throws
        @test_throws AssertionError Extremes.quantile(fm, -1)

        # Test with known values
        @test all(quantile(fm, .99) .≈ quantile.(pd,.99))
    end

    @testset "returnlevel(fm, returnPeriod)" begin
        # returnPeriod < 0 throws
        @test_throws AssertionError Extremes.returnlevel(fm, -1)

        # Test with known values
        @test all(quantile(fm, .99) .≈ quantile.(pd,.99))
    end

    @testset "hessian(fm)" begin
        # Tested in cint(fm)
    end

    @testset "parametervar(fm)" begin
        # Tested in cint(fm)
    end

    @testset "cint(fm)" begin

        @test_throws AssertionError cint(fm, 1)

        c = cint(fm)

        @test all(isapprox.(c[1], [0.8775, 1.1225], atol=.0001))
        @test all(isapprox.(c[2], [0.9087, 1.0913], atol=.0001))
        @test all(isapprox.(c[3], [-.6505, -.3495], atol=.0001))
        @test all(isapprox.(c[4], [0.5624, 1.4376], atol=.0001))
        @test all(isapprox.(c[5], [-.1410, 0.1430], atol=.0001))
        @test all(isapprox.(c[6], [-1.2064, 1.2064], atol=.0001))

    end


    @testset "quantile(AbstractExtremeValueModelr(rl, level)" begin
        # Tested in cint(rl)
    end

    @testset "cint(rl, confidencelevel)" begin
        # confidencelevel not in [0, 1] throws
        @test_throws AssertionError cint(fm, 1)

        # Test with known values

        rl = returnlevel(fm, 100)

        c = cint(rl)

        @test all(isapprox.(c[1], [3.0385, 4.4428], atol=.0001))
        @test all(isapprox.(c[2], [3.1022, 5.5488], atol=.0001))

    end


    df = CSV.read("dataset/gp_nonstationary.csv", DataFrame)

    deleteat!(df, 101:nrow(df))

    n = nrow(df)
    nobservation = 1000
    nobsperblock = 1

    model = Extremes.ThresholdExceedance(Variable("y", df.y),
        logscalecov = [Variable("x₁", df.x₁)])

    θ = [-.5, 1., 0.001]

    k = 3

    fm = MaximumLikelihoodAbstractExtremeValueModel(model, θ)

    ϕ = θ[1] .+ θ[2].*df.x₁
    ξ = θ[3]

    pd = GeneralizedPareto.(exp.(ϕ), ξ)


    @testset "getdistribution(fittedmodel)" begin
        @test all(Extremes.getdistribution(fm) .== pd)
    end

    @testset "loglike(fd)" begin
        # Test with known values
        @test Extremes.loglike(fm) ≈ sum(logpdf.(pd, df.y))
    end

    @testset "quantile(fm, p)" begin
        # p not in [0, 1] throws
        @test_throws AssertionError Extremes.quantile(fm, -1)

        # Test with known values
        @test all(quantile(fm, .99) .≈ quantile.(pd,.99))
    end

    @testset "returnlevel(fm, returnPeriod)" begin
        # returnPeriod < 0 throws
        @test_throws AssertionError returnlevel(fm, 0, nobservation, nobsperblock, -100)

        # Test with known values
        rl = returnlevel(fm, 0, nobservation, nobsperblock, 100)
        p = 1-nobservation/(100 * nobsperblock * n)
        @test all(rl.value .≈ quantile.(pd, p))
    end

    @testset "hessian(fm)" begin
        # Tested in cint(fm)
    end

    @testset "parametervar(fm)" begin
        # Tested in cint(fm)
    end

    @testset "cint(fm)" begin

        @test_throws AssertionError cint(fm, 1)

        # Test with known values
        # Evaluating the confidence interval empirical level using a Monte Carlo approach.
        # Use of the hessian(fm) and parametervar function.

        c = cint(fm)

        @test all(isapprox.(c[1], [-.8077, -.1923], atol=.0001))
        @test all(isapprox.(c[2], [0.3828, 1.6172], atol=.0001))
        @test all(isapprox.(c[3], [-.2812, 0.2832], atol=.0001))

    end


    @testset "quantile(AbstractExtremeValueModelr(rl, level)" begin
        # Tested in cint(rl)
    end

    @testset "cint(rl, confidencelevel)" begin
        # confidencelevel not in [0, 1] throws
        @test_throws AssertionError cint(fm, 1)

        rl = returnlevel(fm, 0, nobservation, nobsperblock, 100)

        c = cint(rl)

        @test all(isapprox.(c[1], [1.0579, 1.7318], atol=.0001))
        @test all(isapprox.(c[2], [1.2515, 2.3947], atol=.0001))

    end


    @testset "showAbstractFittedExtremeValueModel(io, obj, prefix)" begin
        # print does not throw
        buffer = IOBuffer()
        @test_logs Extremes.showAbstractFittedExtremeValueModel(buffer, fm, prefix = "\t")
    end


end
