
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


    @testset "getcluster(y, u₁, u₂)" begin

        # Test with known values
        y = zeros(Int64,10)
        y[1:3] .= 1
        y[5] = 1
        y[9:10] .= 1

        # assertion error
        @test_throws AssertionError getcluster(y,.5,1.0)

        cluster = getcluster(y,.5,0.0)

        @test cluster[1].position == collect(1:3)
        @test cluster[1].value == ones(Int64,3)

        @test cluster[2].position == collect(5:5)
        @test cluster[2].value == ones(Int64,1)

        @test cluster[3].position == collect(9:10)
        @test cluster[3].value == ones(Int64,2)
    end

end
