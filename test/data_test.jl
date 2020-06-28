@testset "data.jl" begin
    @testset "load(name)" begin
        # nonexistent file throws
        @test_throws ErrorException load("nonexistant")

        # portpirie loading
        df = load("portpirie")
        @test size(df, 1) == 65
        @test size(df, 2) == 2
    end

end
