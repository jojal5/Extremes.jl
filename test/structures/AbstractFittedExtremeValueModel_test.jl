@testset "AbstractFittedExtremeValueModel.jl" begin

    @testset "location(::AbstractFittedExtremeValueModel)" begin
        @testset "stationary block maxima" begin
            
            model = Extremes.BlockMaxima{GeneralizedExtremeValue}(Variable("y", [1]))
            θ̂ = [0., 0., 0.]
            s = MaximumLikelihoodAbstractExtremeValueModel(model, θ̂)
            
            @test all(location(s) .≈ θ̂[1])
            
        end
        
        @testset "non-stationary block maxima" begin
            
            model  = BlockMaxima{GeneralizedExtremeValue}(Variable("y", [1, 2]), locationcov = [Variable("x", [0,1])])
            θ̂ = [0., 1., 0., 0.]
            s = MaximumLikelihoodAbstractExtremeValueModel(model, θ̂)
            
            @test all(location(s) .≈ [0., 1.])
            
        end
        
        @testset "stationary POT" begin
            
            model = Extremes.ThresholdExceedance(Variable("y", [1]))
            θ̂ = [0., 0.]
            s = MaximumLikelihoodAbstractExtremeValueModel(model, θ̂)
            
            @test all(location(s) .≈ 0.)
            
        end
        
        @testset "non-stationary POT" begin
            
            model = Extremes.ThresholdExceedance(Variable("y", [1, 2]), logscalecov = [Variable("x", [0, 1])])
            θ̂ = [0., 0., 0.]
            s = MaximumLikelihoodAbstractExtremeValueModel(model, θ̂)
            
            @test all(location(s) .≈ 0.)
            
        end
        
    end

    @testset "params(AbstractFittedExtremeValueModel)" begin
        @testset "stationary block maxima" begin
        
            model = Extremes.BlockMaxima{GeneralizedExtremeValue}(Variable("y", [1]))
            θ̂ = [0., 0., 0.]
            s = MaximumLikelihoodAbstractExtremeValueModel(model, θ̂)
            
            @test all(params(s)[] .≈ (0.,1.,0.))
            
        end
        
        @testset "non-stationary block maxima" begin
            
            model = model  = BlockMaxima{GeneralizedExtremeValue}(Variable("y", [1]), locationcov = [Variable("x", [1])])
            θ̂ = [0., 0., 0., 0.]
            s = MaximumLikelihoodAbstractExtremeValueModel(model, θ̂)
            
            @test all(params(s)[] .≈ (0.,1.,0.))
            
        end
        
        @testset "stationary POT" begin
            
            model = Extremes.ThresholdExceedance(Variable("y", [1]))
            θ̂ = [0., 0.]
            s = MaximumLikelihoodAbstractExtremeValueModel(model, θ̂)
            
            @test all(params(s)[] .≈ (0.,1.,0.))
            
        end
        
        @testset "non-stationary POT" begin
            
            model = Extremes.ThresholdExceedance(Variable("y", [1]), logscalecov = [Variable("x", [1])])
            θ̂ = [0., 0., 0.]
            s = MaximumLikelihoodAbstractExtremeValueModel(model, θ̂)
            
            @test all(params(s)[] .≈ (0.,1.,0.))
            
        end
    end

    @testset "scale(::AbstractFittedExtremeValueModel)" begin

        @testset "stationary block maxima" begin
            
            model = Extremes.BlockMaxima{GeneralizedExtremeValue}(Variable("y", [1]))
            θ̂ = [0., 0., 0.]
            s = MaximumLikelihoodAbstractExtremeValueModel(model, θ̂)
            
            @test all(scale(s) .≈ 1.)
            
        end
        
        @testset "non-stationary block maxima" begin
            
            model  = BlockMaxima{GeneralizedExtremeValue}(Variable("y", [1, 2]), logscalecov = [Variable("x", [0,1])])
            θ̂ = [0., 0, 1., 0.]
            s = MaximumLikelihoodAbstractExtremeValueModel(model, θ̂)
            
            @test all(scale(s) .≈ exp.([0., 1.]))
            
        end
        
        @testset "stationary POT" begin
            
            model = Extremes.ThresholdExceedance(Variable("y", [1]))
            θ̂ = [0., 0.]
            s = MaximumLikelihoodAbstractExtremeValueModel(model, θ̂)
            
            @test all(scale(s) .≈ 1.)
            
        end
        
        @testset "non-stationary POT" begin
            
            model = Extremes.ThresholdExceedance(Variable("y", [1, 2]), logscalecov = [Variable("x", [0, 1])])
            θ̂ = [0., 1., 0.]
            s = MaximumLikelihoodAbstractExtremeValueModel(model, θ̂)
            
            @test all(scale(s) .≈ exp.([0., 1.]))
            
        end
        
    end

    @testset "shape(::AbstractFittedExtremeValueModel)" begin

        @testset "stationary block maxima" begin
            
            model = Extremes.BlockMaxima{GeneralizedExtremeValue}(Variable("y", [1]))
            θ̂ = [0., 0., 0.]
            s = MaximumLikelihoodAbstractExtremeValueModel(model, θ̂)
            
            @test all(shape(s) .≈ 0.)
            
        end
        
        @testset "non-stationary block maxima" begin
            
            model  = BlockMaxima{GeneralizedExtremeValue}(Variable("y", [1, 2]), shapecov = [Variable("x", [0,1])])
            θ̂ = [0., 0, 0., 1.]
            s = MaximumLikelihoodAbstractExtremeValueModel(model, θ̂)
            
            @test all(shape(s) .≈ [0., 1.])
            
        end
        
        @testset "stationary POT" begin
            
            model = Extremes.ThresholdExceedance(Variable("y", [1]))
            θ̂ = [0., 0.]
            s = MaximumLikelihoodAbstractExtremeValueModel(model, θ̂)
            
            @test all(shape(s) .≈ 0.)
            
        end
        
        @testset "non-stationary POT" begin
            
            model = Extremes.ThresholdExceedance(Variable("y", [1, 2]), shapecov = [Variable("x", [0, 1])])
            θ̂ = [0., 0., 1.]
            s = MaximumLikelihoodAbstractExtremeValueModel(model, θ̂)
            
            @test all(shape(s) .≈ [0., 1.])
            
        end
        
    end

    @testset "Base.show(io, obj)" begin
        # Print BayesianAbstractExtremeValueModel does not throw
        fm = Extremes.BayesianAbstractExtremeValueModel(Extremes.BlockMaxima{GeneralizedExtremeValue}(Variable("y", [100.0])), MambaLite.Chains([100.0 log(5.0) .1]))
        buffer = IOBuffer()
        @test_logs Base.show(buffer, fm)

        # Print MaximumLikelihoodAbstractExtremeValueModel does not throw
        fm = Extremes.MaximumLikelihoodAbstractExtremeValueModel(BlockMaxima{GeneralizedExtremeValue}(Variable("y", [1])), [1.0, 1.0, 0.1])
        @test_logs Base.show(buffer, fm)

        # Print PwmAbstractExtremeValueModel does not throw
        fm = Extremes.pwmAbstractExtremeValueModel{BlockMaxima{GeneralizedExtremeValue}}(Extremes.BlockMaxima{GeneralizedExtremeValue}(Variable("y", [0])), [1.0; 0.0; 0.1])
        @test_logs Base.show(buffer, fm)

    end

    include(joinpath("AbstractFittedExtremeValueModel", "bayesianAbstractExtremeValueModel_test.jl"))
    include(joinpath("AbstractFittedExtremeValueModel", "maximumlikelihoodAbstractExtremeValueModel_test.jl"))
    include(joinpath("AbstractFittedExtremeValueModel", "pwmAbstractExtremeValueModel_test.jl"))

end
