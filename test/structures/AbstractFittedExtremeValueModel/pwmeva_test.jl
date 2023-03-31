@testset "pwmeva.jl" begin

    θ = [0.0, 0.0, 0.1]

    pd = GeneralizedExtremeValue(θ[1], exp(θ[2]), θ[3])

    y = [0]

    fm = Extremes.pwmEVA{BlockMaxima{GeneralizedExtremeValue}}(Extremes.BlockMaxima{GeneralizedExtremeValue}(Variable("y", y)), θ)

    @testset "getdistribution(fittedmodel)" begin
        @test Extremes.getdistribution(fm)[] == pd
    end

    @testset "quantile(fm, p)" begin
        # p outside of [0, 1] throws
        @test_throws AssertionError Extremes.quantile(fm, -1)

        # Test with known values
        @test quantile(fm, .99)[] ≈ quantile(pd,.99)

    end

    @testset "returnlevel(fm, returnPeriod)" begin

        # returnPeriod < 0 throws
        @test_throws AssertionError returnlevel(fm, -1)

        # Test with known values
        @test returnlevel(fm, 100).value[] ≈ quantile(pd, 1-1/100)
    end

    @testset "parametervar(fm, nboot)" begin
        # nboot < 0 throws
        @test_throws AssertionError Extremes.parametervar(fm, -1)

        # Tested in cint(fm)
    end

    @testset "quantilevar(fm, level, nboot)" begin
        # Tested in cint(fm)
    end

    @testset "cint(fm)" begin
        # returnPeriod < 0 throws
        @test_throws AssertionError cint(ReturnLevel(Extremes.BlockMaximaModel(fm), -1, [1.0]), 0.95)

        # confidencelevel not in [0, 1]
        @test_throws AssertionError cint(ReturnLevel(Extremes.BlockMaximaModel(fm), 1, [1.0]), -1)

        niter = 1000

        cover = falses(niter,3)

        Threads.@threads  for i in 1:niter
            y = rand(pd, 100)
            fm = gevfitpwm(y)
            ci = cint(fm, .95 ,1000)
            for j in eachindex(θ)
                cover[i,j] = ci[j][1] < θ[j] < ci[j][2]
            end
        end

        @test all(count.(eachcol(cover))/niter .> .85)
    end





    @testset "cint(fm, returnPeriod, confidencelevel)" begin
        # confidencelevel not in [0, 1]
        @test_throws AssertionError cint(ReturnLevel(Extremes.BlockMaximaModel(fm), 1, [1.0]), -1)

        # Test with known values

        θ = [100; log(5); .1]

        pd = GeneralizedExtremeValue(θ[1],exp(θ[2]),θ[3])

        q = quantile(pd, 1-1/100)

        niter = 1200

        cover = falses(niter)

        Threads.@threads  for i in 1:niter
            y = rand(pd, 200)
            fm = gevfitpwm(y)
            rl = returnlevel(fm, 100)
            ci = cint(rl, .95 , 500)
            cover[i] = ci[][1] < q < ci[][2]
        end

        @test count(cover)/niter > .85
    end


    @testset "showAbstractFittedExtremeValueModel(io, obj, prefix)" begin
        # print does not throw
        buffer = IOBuffer()
        @test_logs Extremes.showAbstractFittedExtremeValueModel(buffer, fm, prefix = "\t")

    end

end
