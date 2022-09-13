@testset "fittedeva.jl" begin
    @testset "Base.show(io, obj)" begin
        # Print BayesianEVA does not throw
        fm = Extremes.BayesianEVA(Extremes.BlockMaxima(Variable("y", [100.0])), MCMCChains.Chains([100.0 log(5.0) .1]))
        buffer = IOBuffer()
        @test_logs Base.show(buffer, fm)

        # Print MaximumLikelihoodEVA does not throw
        fm = Extremes.MaximumLikelihoodEVA(BlockMaxima(Variable("y", [1])), [1.0, 1.0, 0.1])
        @test_logs Base.show(buffer, fm)

        # Print PwmEVA does not throw
        fm = Extremes.pwmEVA{BlockMaxima, GeneralizedExtremeValue}(Extremes.BlockMaxima(Variable("y", [0])), [1.0; 0.0; 0.1])
        @test_logs Base.show(buffer, fm)

    end

    include(joinpath("fittedeva", "bayesianeva_test.jl"))
    include(joinpath("fittedeva", "maximumlikelihoodeva_test.jl"))
    include(joinpath("fittedeva", "pwmeva_test.jl"))

end