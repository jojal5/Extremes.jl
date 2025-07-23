@testset "standarddist" begin
    model1 = Extremes.BlockMaxima{GeneralizedExtremeValue}(Variable("y", [1]))
    model2 = Extremes.BlockMaxima{Gumbel}(Variable("y", [2]))
    @test Extremes.standarddist(model1)==Gumbel()
    @test Extremes.standarddist(model2)==Gumbel()
end

include(joinpath("blockmaxima", "blockmaxima{GeneralizedExtremeValue}_test.jl"))
include(joinpath("blockmaxima", "blockmaxima{Gumbel}_test.jl"))