@testset "dataitem.jl" begin
    @testset "Standardization of the Variable type" begin

        v = Variable("x", 100 .+ 5*randn(100))

        vstd = Extremes.standardize(v)

        m = mean(vstd.value)
        s = std(vstd.value)

        @test m ≈ 0.0 atol = 1e-10
        @test s ≈ 1.0 atol = 1e-10
        @test vstd.offset ≈ 100 atol = 5
        @test vstd.scale ≈ 5 atol = .5

        ṽ = Extremes.reconstruct(vstd)

        @test ṽ.name == v.name
        @test ṽ.value ≈ v.value atol = 1e-10

        v = Variable("x", [-1; 0 ; 1])
        vstd = Extremes.standardize(v)

        @test vstd.offset == 0 # integer comparison
        @test vstd.scale == 1 # integer comparison

        v = Variable("x", ones(10))
        vstd = Extremes.standardize(v)

        @test vstd.scale == 1 # integer comparison

    end

end
