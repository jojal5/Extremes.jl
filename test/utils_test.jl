@testset "utils.jl" begin
    
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
