@testset "fittedeva.jl" begin

    @testset "location(::fittedEVA)" begin
        @testset "stationary block maxima" begin
            
            model = Extremes.BlockMaxima{GeneralizedExtremeValue}(Variable("y", [1]))
            θ̂ = [0., 0., 0.]
            s = MaximumLikelihoodEVA(model, θ̂)
            
            @test all(location(s) .≈ θ̂[1])
            
        end
        
        @testset "non-stationary block maxima" begin
            
            model  = BlockMaxima{GeneralizedExtremeValue}(Variable("y", [1, 2]), locationcov = [Variable("x", [0,1])])
            θ̂ = [0., 1., 0., 0.]
            s = MaximumLikelihoodEVA(model, θ̂)
            
            @test all(location(s) .≈ [0., 1.])
            
        end
        
        @testset "stationary POT" begin
            
            model = Extremes.ThresholdExceedance(Variable("y", [1]))
            θ̂ = [0., 0.]
            s = MaximumLikelihoodEVA(model, θ̂)
            
            @test all(location(s) .≈ 0.)
            
        end
        
        @testset "non-stationary POT" begin
            
            model = Extremes.ThresholdExceedance(Variable("y", [1, 2]), logscalecov = [Variable("x", [0, 1])])
            θ̂ = [0., 0., 0.]
            s = MaximumLikelihoodEVA(model, θ̂)
            
            @test all(location(s) .≈ 0.)
            
        end
        
    end

    @testset "params(fittedEVA)" begin
        @testset "stationary block maxima" begin
        
            model = Extremes.BlockMaxima{GeneralizedExtremeValue}(Variable("y", [1]))
            θ̂ = [0., 0., 0.]
            s = MaximumLikelihoodEVA(model, θ̂)
            
            @test all(params(s) .≈ θ̂)
            
        end
        
        @testset "non-stationary block maxima" begin
            
            model = model  = BlockMaxima{GeneralizedExtremeValue}(Variable("y", [1]), locationcov = [Variable("x", [1])])
            θ̂ = [0., 0., 0., 0.]
            s = MaximumLikelihoodEVA(model, θ̂)
            
            @test all(params(s) .≈ θ̂)
            
        end
        
        @testset "stationary POT" begin
            
            model = Extremes.ThresholdExceedance(Variable("y", [1]))
            θ̂ = [0., 0.]
            s = MaximumLikelihoodEVA(model, θ̂)
            
            @test all(params(s) .≈ θ̂)
            
        end
        
        @testset "non-stationary POT" begin
            
            model = Extremes.ThresholdExceedance(Variable("y", [1]), logscalecov = [Variable("x", [1])])
            θ̂ = [0., 0., 0.]
            s = MaximumLikelihoodEVA(model, θ̂)
            
            @test all(params(s) .≈ θ̂)
            
        end
    end

    @testset "scale(::fittedEVA)" begin

        @testset "stationary block maxima" begin
            
            model = Extremes.BlockMaxima{GeneralizedExtremeValue}(Variable("y", [1]))
            θ̂ = [0., 0., 0.]
            s = MaximumLikelihoodEVA(model, θ̂)
            
            @test all(scale(s) .≈ 1.)
            
        end
        
        @testset "non-stationary block maxima" begin
            
            model  = BlockMaxima{GeneralizedExtremeValue}(Variable("y", [1, 2]), logscalecov = [Variable("x", [0,1])])
            θ̂ = [0., 0, 1., 0.]
            s = MaximumLikelihoodEVA(model, θ̂)
            
            @test all(scale(s) .≈ exp.([0., 1.]))
            
        end
        
        @testset "stationary POT" begin
            
            model = Extremes.ThresholdExceedance(Variable("y", [1]))
            θ̂ = [0., 0.]
            s = MaximumLikelihoodEVA(model, θ̂)
            
            @test all(scale(s) .≈ 1.)
            
        end
        
        @testset "non-stationary POT" begin
            
            model = Extremes.ThresholdExceedance(Variable("y", [1, 2]), logscalecov = [Variable("x", [0, 1])])
            θ̂ = [0., 1., 0.]
            s = MaximumLikelihoodEVA(model, θ̂)
            
            @test all(scale(s) .≈ exp.([0., 1.]))
            
        end
        
    end

    @testset "shape(::fittedEVA)" begin

        @testset "stationary block maxima" begin
            
            model = Extremes.BlockMaxima{GeneralizedExtremeValue}(Variable("y", [1]))
            θ̂ = [0., 0., 0.]
            s = MaximumLikelihoodEVA(model, θ̂)
            
            @test all(shape(s) .≈ 0.)
            
        end
        
        @testset "non-stationary block maxima" begin
            
            model  = BlockMaxima{GeneralizedExtremeValue}(Variable("y", [1, 2]), shapecov = [Variable("x", [0,1])])
            θ̂ = [0., 0, 0., 1.]
            s = MaximumLikelihoodEVA(model, θ̂)
            
            @test all(shape(s) .≈ [0., 1.])
            
        end
        
        @testset "stationary POT" begin
            
            model = Extremes.ThresholdExceedance(Variable("y", [1]))
            θ̂ = [0., 0.]
            s = MaximumLikelihoodEVA(model, θ̂)
            
            @test all(shape(s) .≈ 0.)
            
        end
        
        @testset "non-stationary POT" begin
            
            model = Extremes.ThresholdExceedance(Variable("y", [1, 2]), shapecov = [Variable("x", [0, 1])])
            θ̂ = [0., 0., 1.]
            s = MaximumLikelihoodEVA(model, θ̂)
            
            @test all(shape(s) .≈ [0., 1.])
            
        end
        
    end

    @testset "Base.show(io, obj)" begin
        # Print BayesianEVA does not throw
        fm = Extremes.BayesianEVA(Extremes.BlockMaxima{GeneralizedExtremeValue}(Variable("y", [100.0])), MambaLite.Chains([100.0 log(5.0) .1]))
        buffer = IOBuffer()
        @test_logs Base.show(buffer, fm)

        # Print MaximumLikelihoodEVA does not throw
        fm = Extremes.MaximumLikelihoodEVA(BlockMaxima{GeneralizedExtremeValue}(Variable("y", [1])), [1.0, 1.0, 0.1])
        @test_logs Base.show(buffer, fm)

        # Print PwmEVA does not throw
        fm = Extremes.pwmEVA{BlockMaxima{GeneralizedExtremeValue}}(Extremes.BlockMaxima{GeneralizedExtremeValue}(Variable("y", [0])), [1.0; 0.0; 0.1])
        @test_logs Base.show(buffer, fm)

    end

    include(joinpath("fittedeva", "bayesianeva_test.jl"))
    include(joinpath("fittedeva", "maximumlikelihoodeva_test.jl"))
    include(joinpath("fittedeva", "pwmeva_test.jl"))

end
