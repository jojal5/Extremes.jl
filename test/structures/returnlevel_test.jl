@testset "returnlevel.jl" begin
    @testset "Base.show(io, obj)" begin
        # Print ReturnLevel does not throw
        fm = MaximumLikelihoodEVA(BlockMaxima(Variable("y", [1])), [1.0, 1.0, 0.1])
        rl = ReturnLevel(fm, 10, [1.0])

        buffer = IOBuffer()
        @test_logs Base.show(buffer, rl)

    end

end
