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

    end

end
