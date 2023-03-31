@testset "returnlevel.jl" begin
    @testset "Base.show(io, obj)" begin
        # Print ReturnLevel does not throw
        fm = MaximumLikelihoodAbstractExtremeValueModel(BlockMaxima{GeneralizedExtremeValue}(Variable("y", [1])), [1.0, 1.0, 0.1])
        rl = ReturnLevel(Extremes.BlockMaximaModel(fm), 10, [1.0])

        buffer = IOBuffer()
        @test_logs Base.show(buffer, rl)

    end

end
