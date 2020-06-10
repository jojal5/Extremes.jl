@testset "utils.jl" begin
    @testset "getcluster(y, μ₁, μ₂)" begin
        # TODO : Add tests with known values?
    end

    @testset "getcluster(df, μ₁, μ₂)" begin
        # TODO : Test with coltype[1] not a date

        # TODO : Test with coltype[2] not a real

        # TODO : Add tests with known values?

    end

    @testset "slicematrix(A; dims)" begin
        # TODO : Test with dims > 2

        # slice a matrix of size 10 x 5
        m = 10
        n = 5
        A = rand(1:10, m, n)

        B = Extremes.slicematrix(A)
        C = Extremes.slicematrix(A, dims = 2)

        Aᴮ = Extremes.unslicematrix(B)
        Aᶜ = Extremes.unslicematrix(C, dims = 2)

        @test length(B) == n
        @test length(C) == m
        @test A == Aᴮ
        @test A == Aᶜ

    end

    @testset "unslicematrix(B; dims)" begin
        # TODO : Test with dims > 2

        # TODO : Test with known values

    end

    @testset "buildExplanatoryVariables(df, ids)" begin
        # TODO : Test with known values
        
    end

end
