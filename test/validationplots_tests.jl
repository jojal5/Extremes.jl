@testset "validationplots.jl" begin
    @testset "ecdf(y)" begin
        n = 10
        data = -1 * collect(1:n)

        y, p = Extremes.ecdf(data)

        # y is length n and sorted
        @test length(y) == n
        @test issorted(y)

        # p is length n, sorted and between 0 and 1.
        @test length(p) == n
        @test issorted(p)
        @test p[1] >= 0 && p[end] <= 1

    end

    include(joinpath("validationplots", "plots_std_tests.jl"))
    include(joinpath("validationplots", "plots_tests.jl"))

end
