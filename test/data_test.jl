@testset "data.jl" begin
    @testset "dataset(name)" begin
        # nonexistent file throws
        @test_throws ErrorException Extremes.dataset("nonexistant")

        # portpirie loading
        df = Extremes.dataset("portpirie")
        @test size(df, 1) == 65
        @test size(df, 2) == 2
    end

end
