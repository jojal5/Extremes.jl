@testset "fittedeva.jl" begin
    @testset "Base.show(io, obj)" begin
        # Print BayesianEVA does not throw
        fm = Extremes.BayesianEVA(Extremes.BlockMaxima(Variable("y", [100.0])), Mamba.Chains([100.0 log(5.0) .1]))
        buffer = IOBuffer()
        @test_logs Base.show(buffer, fm)

        # Print MaximumLikelihoodEVA does not throw
        fm = Extremes.MaximumLikelihoodEVA(BlockMaxima(Variable("y", [1])), [1.0, 1.0, 0.1])
        @test_logs Base.show(buffer, fm)

        # Print PwmEVA does not throw
        fm = Extremes.pwmEVA(Extremes.BlockMaxima(Variable("y", [0])), [1.0; 0.0; 0.1])
        @test_logs Base.show(buffer, fm)

    end

    @testset "Base.show(io, obj)" begin
        # Print ReturnLevel does not throw
        fmb = Extremes.BayesianEVA(Extremes.BlockMaxima(Variable("y", [100.0])), Mamba.Chains([100.0 log(5.0) .1]))
        rl = ReturnLevel(fmb, 10, [1.0], [[0.9, 1.1]])
        buffer = IOBuffer()
        @test_logs Base.show(buffer, rl)

    end

    include(joinpath("fittedeva", "bayesianeva_test.jl"))
    include(joinpath("fittedeva", "maximumlikelihoodeva_test.jl"))
    include(joinpath("fittedeva", "pwmeva_test.jl"))

end
