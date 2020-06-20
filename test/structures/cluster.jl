
@testset "cluster.jl" begin
    @testset "constructor" begin
        # assertion error
        @test_throws AssertionError Cluster(10,11,[51],[85])
        @test_throws AssertionError Cluster(10,0,[51],[85; 85])

        # Test with known values
        c = Cluster(10,0,[51],[85])
        @test c.u₁ ≈ 10
        @test c.u₂ ≈ 0
        @test c.position ≈ [51]
        @test c.value ≈ [85]
    end

end
