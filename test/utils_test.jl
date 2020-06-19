@testset "utils.jl" begin
    @testset "getcluster(y, μ₁, μ₂)" begin
        # TODO : Add tests with known values (J)
    end

    @testset "getcluster(df, μ₁, μ₂)" begin
        # coltype[1] not a date throws
        df = DataFrame(a = [1.0], b = [1.0])
        @test_throws AssertionError Extremes.getcluster(df, 1.0)

        # coltype[2] not a real throws
        df = DataFrame(a = [Date(2020)], b = [Date(2020)])
        @test_throws AssertionError Extremes.getcluster(df, 1.0)

        # TODO : Add tests with known values (J)

    end

    @testset "slicematrix(A; dims)" begin
        m = 10
        n = 5
        A = rand(1:10, m, n)

        # dims > 2 throws
        @test_throws AssertionError Extremes.slicematrix(A, dims = 3)

        # slice a matrix of size 10 x 5 (dim = 1)
        B = Extremes.slicematrix(A)

        @test length(B) == n

        # slice a matrix of size 10 x 5 (dim = 1)
        C = Extremes.slicematrix(A, dims = 2)

        @test length(C) == m

    end

    @testset "unslicematrix(B; dims)" begin
        m = 10
        n = 5
        A = rand(1:10, m, n)
        B = Extremes.slicematrix(A)

        # dims > 2 throws
        @test_throws AssertionError Extremes.unslicematrix(B, dims = 3)

        # unslice a matrix of size 10 x 5 (dim = 1)
        Aᴮ = Extremes.unslicematrix(B)

        @test A == Aᴮ

        # unslice a matrix of size 10 x 5 (dim = 1)
        C = Extremes.slicematrix(A, dims = 2)
        Aᶜ = Extremes.unslicematrix(C, dims = 2)

        @test A == Aᶜ

    end

    @testset "buildVariables(df, ids)" begin
        val = rand(10)
        name = "x"

        df = DataFrame(x = val)

        # stationary
        evs = Extremes.buildVariables(df, Symbol[])
        @test length(evs) == 0

        # non-stationary
        evs = Extremes.buildVariables(df, [:x])

        @test length(evs) == 1
        @test evs[1].name == name
        @test evs[1].value == val

    end

end
