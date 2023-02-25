@testset "maximumlikelihoodeva.jl" begin

    n = 1000

    x₁ = Variable("x₁", randn(n)/3)
    x₂ = Variable("x₂", randn(n))
    x₃ = Variable("x₂", randn(n))

    θ = [1; 1; 1; -.5; 1; .1]

    μ = θ[1] .+ x₂.value*θ[2] .+ x₃.value*θ[3]
    ϕ = θ[4] .+ x₁.value*θ[5]
    ξ = θ[6]

    pd = GeneralizedExtremeValue.(μ, exp.(ϕ), ξ)

    y = rand.(pd)

    model = BlockMaxima{GeneralizedExtremeValue}(Variable("y", y), locationcov=[x₂; x₃], logscalecov = [x₁])

    fm = Extremes.MaximumLikelihoodEVA(model, θ)


    @testset "getdistribution(fittedmodel)" begin
        @test all(Extremes.getdistribution(fm) .== pd)
    end

    @testset "loglike(fd)" begin
        # Test with known values
        @test Extremes.loglike(fm) ≈ sum(logpdf.(pd,y))
    end

    @testset "quantile(fm, p)" begin
        # p not in [0, 1] throws
        @test_throws AssertionError Extremes.quantile(fm, -1)

        # Test with known values
        @test quantile(fm, .99) ≈ quantile.(pd,.99)
    end

    @testset "returnlevel(fm, returnPeriod)" begin
        # returnPeriod < 0 throws
        @test_throws AssertionError Extremes.returnlevel(fm, -1)

        # Test with known values
        @test quantile(fm, .99) ≈ quantile.(pd,.99)
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

        niter = 1000

        cover = falses(niter,6)

        Threads.@threads  for i in 1:niter
            y = rand.(pd)
            m = BlockMaxima{GeneralizedExtremeValue}(Variable("y",y), locationcov = [x₂, x₃], logscalecov=[x₁])
            fm = gevfit(m, θ)
            ci = cint(fm)
            for j in eachindex(θ)
                cover[i,j] = ci[j][1] < θ[j] < ci[j][2]
            end
        end

        @test all(count.(eachcol(cover))/niter .> .9)

    end


    @testset "quantilevar(rl, level)" begin
        # Tested in cint(rl)
    end

    @testset "cint(rl, confidencelevel)" begin
        # confidencelevel not in [0, 1] throws
        @test_throws AssertionError cint(fm, 1)

        # Test with known values

        θ = [100; log(5); .1]

        pd = GeneralizedExtremeValue(θ[1],exp(θ[2]),θ[3])

        q = quantile(pd, 1-1/100)

        niter = 1000

        cover = falses(niter)

        Threads.@threads  for i in 1:niter
            y = rand(pd, 300)
            m = BlockMaxima{GeneralizedExtremeValue}(Variable("y",y))
            fm = gevfit(m, θ)
            rl = returnlevel(fm, 100)
            ci = cint(rl)
            cover[i] = ci[][1] < q < ci[][2]
        end

        @test count(cover)/niter .> .9

    end

    n = 1000
    nobservation = 10000
    nobsperblock = 1

    x = Variable("x",randn(n))

    θ = [-.5; 1; 0.1]

    ϕ = θ[1] .+ θ[2]*x.value
    σ = exp.(ϕ)
    ξ = θ[3]

    pd = GeneralizedPareto.(σ, ξ)

    y = rand.(pd)

    model = ThresholdExceedance(Variable("y", y), logscalecov = [x])

    fm = Extremes.MaximumLikelihoodEVA(model, θ)

    @testset "getdistribution(fittedmodel)" begin
        @test all(Extremes.getdistribution(fm) .== pd)
    end

    @testset "loglike(fd)" begin
        # Test with known values
        @test Extremes.loglike(fm) ≈ sum(logpdf.(pd,y))
    end

    @testset "quantile(fm, p)" begin
        # p not in [0, 1] throws
        @test_throws AssertionError Extremes.quantile(fm, -1)

        # Test with known values
        @test quantile(fm, .99) ≈ quantile.(pd,.99)
    end

    @testset "returnlevel(fm, returnPeriod)" begin
        # returnPeriod < 0 throws
        @test_throws AssertionError returnlevel(fm, 0, nobservation, nobsperblock, -100)

        # Test with known values
        rl = returnlevel(fm, 0, nobservation, nobsperblock, 100)
        p = 1-nobservation/(100 * nobsperblock * n)
        @test rl.value ≈ quantile.(pd, p)
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

        niter = 1000

        cover = falses(niter,length(θ))

        Threads.@threads  for i in 1:niter
            y = rand.(pd)
            m = ThresholdExceedance(Variable("y",y), logscalecov=[x])
            fm = gpfit(m, θ)
            ci = cint(fm)
            for j in eachindex(θ)
                cover[i,j] = ci[j][1] < θ[j] < ci[j][2]
            end
        end

        @test all(count.(eachcol(cover))/niter .> .9)

    end


    @testset "quantilevar(rl, level)" begin
        # Tested in cint(rl)
    end

    @testset "cint(rl, confidencelevel)" begin
        # confidencelevel not in [0, 1] throws
        @test_throws AssertionError cint(fm, 1)

        # Test with known values

        θ = [log(5); .1]

        pd = GeneralizedPareto(exp(θ[1]),θ[2])

        n = 100
        nobservation = 1000
        nobsperblock = 1

        p = 1-nobservation/(100 * nobsperblock * n)

        q = quantile(pd, p)

        niter = 1000

        cover = falses(niter)

        Threads.@threads  for i in 1:niter
            y = rand(pd, n)
            m = ThresholdExceedance(Variable("y",y))
            fm = gpfit(m, θ)
            rl = returnlevel(fm, 0, nobservation, nobsperblock, 100)
            ci = cint(rl)
            cover[i] = ci[][1] < q < ci[][2]
        end

        @test count(cover)/niter .> .9

    end


    @testset "showfittedEVA(io, obj, prefix)" begin
        # print does not throw
        buffer = IOBuffer()
        @test_logs Extremes.showfittedEVA(buffer, fm, prefix = "\t")
    end


end
