@testset "utils.jl" begin
    @testset "getcluster(y, μ₁, μ₂)" begin

    end

    @testset "getcluster(df, μ₁, μ₂)" begin

    end

    @testset "slicematrix(A; dims)" begin
        # TODO : Description here
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

    end

    @testset "buildExplanatoryVariables(df, ids)" begin

    end

end
