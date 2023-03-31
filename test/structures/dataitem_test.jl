@testset "dataitem.jl" begin
    @testset "Standardization of the Variable type" begin

        v = Variable("x", 100 .+ 5*randn(100))

        vstd = Extremes.standardize(v)

        m = mean(vstd.value)
        s = std(vstd.value)

        @test m ≈ 0.0 atol = sqrt(eps())
        @test s ≈ 1.0 atol = sqrt(eps())
        @test vstd.offset ≈ 100 atol = 5
        @test vstd.scale ≈ 5 atol = 1

        ṽ = Extremes.reconstruct(vstd)

        @test ṽ.name == v.name
        @test ṽ.value ≈ v.value atol = sqrt(eps())

        v = Variable("x", [-1; 0 ; 1])
        vstd = Extremes.standardize(v)

        @test vstd.offset == 0 # integer comparison
        @test vstd.scale == 1 # integer comparison

        v = Variable("x", ones(10))
        vstd = Extremes.standardize(v)

        @test vstd.scale == 1 # integer comparison

        vstd = VariableStd("x", [-1; 0 ; 1])
        @test vstd.offset == 0 # integer comparison
        @test vstd.scale == 1 # integer comparison

        @test_throws AssertionError Extremes.VariableStd("x", [-2; 0 ; 2])
        @test_throws AssertionError Extremes.VariableStd("x", [0; 1 ; 2])

        # Test on the transform function
        y = randn(3)
        x₁ = Extremes.standardize(Variable("x₁", [-1; 0; 1]))
        x₂ = Extremes.standardize(Variable("x₂", [0; 1; 2]))
        x₃ = Extremes.standardize(Variable("x₃", [0; 2; 4]))

        model = BlockMaxima{GeneralizedExtremeValue}(Variable("y", y), locationcov=[x₁; x₂], logscalecov = [x₃])

        θ̃ = collect(1.0:6.0)

        fm_std = MaximumLikelihoodAbstractExtremeValueModel(model, θ̃)

        fm = Extremes.transform(fm_std)

        θ̂ = fm.θ̂

        @test θ̂[1] ≈ θ̃[1] - θ̃[2]*0/1 - θ̃[3]*1/1
        @test θ̂[2] ≈ θ̃[2]/1
        @test θ̂[3] ≈ θ̃[3]/1
        @test θ̂[4] ≈ θ̃[4] - θ̃[5]*2/2
        @test θ̂[5] ≈ θ̃[5]/2
        @test θ̂[6] ≈ θ̃[6]

    end

end
