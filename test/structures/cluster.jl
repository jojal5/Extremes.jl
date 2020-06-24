
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

    @testset "getcluster(y, u)" begin

        # Test with known values
        y = zeros(Int64,10)
        y[1:3] .= 1
        y[5] = 1
        y[9:10] .= 1

        cluster = getcluster(y,.5)

        @test cluster[1].position == collect(1:3)
        @test cluster[1].value == ones(Int64,3)

        @test cluster[2].position == collect(5:5)
        @test cluster[2].value == ones(Int64,1)

        @test cluster[3].position == collect(9:10)
        @test cluster[3].value == ones(Int64,2)
    end

    @testset "length(c::Cluster)" begin

        # Test with known values
        y = zeros(Int64,10)
        y[1:3] .= 1
        y[5] = 1
        y[9:10] .= 1

        cluster = getcluster(y,.5)

        @test length.(cluster) == [3; 1; 2]
    end

    @testset "max(c::Cluster)" begin

        # Test with known values
        y = zeros(Int64,10)
        y[1] = 5
        y[2] = 2
        y[3] = 1
        y[5] = 3
        y[9:10] .= 1

        cluster = getcluster(y,.5)

        @test maximum.(cluster) == [5; 3; 1]
    end

    @testset "sum(c::Cluster)" begin

        # Test with known values
        y = zeros(Int64,10)
        y[1:3] .= 1
        y[5] = 1
        y[9:10] .= 1

        cluster = getcluster(y,.5)

        @test sum.(cluster) == [3; 1; 2]
    end

    @testset "merge(c₁::Cluster, c₂::Cluster)" begin

        # Test assertion
        c₁ = Cluster(1,0,[10],[100])
        c₂ = Cluster(0,0,[10],[100])
        c₃ = Cluster(1,1,[10],[100])

        @test_throws AssertionError Extremes.merge(c₁, c₂)
        @test_throws AssertionError Extremes.merge(c₁, c₃)

        # Test with known values
        y = zeros(Int64,10)
        y[1:3] .= 1
        y[5] = 1
        y[9:10] .= 1

        cluster = getcluster(y,.5)

        c = Extremes.merge(cluster[1], cluster[2])

        @test c.u₁ ≈ cluster[1].u₁
        @test c.u₂ ≈ cluster[1].u₁
        @test c.position == [1;2;3;5]
        @test c.value == [1;1;1;1]
    end

end
