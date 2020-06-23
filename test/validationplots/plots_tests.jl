@testset "plots.jl" begin
    y = collect(1:5)
    pd = GeneralizedExtremeValue(1.0, 1.0, 0.1)

    fm = MaximumLikelihoodEVA(BlockMaxima(Variable("y", y)), [1.0, 0.0, 0.1])

    @testset "probplot(fm)" begin
        # Plot does not throw
        @test_logs Extremes.probplot(fm)

    end

    @testset " qqplot(fm)" begin
        # Plot does not throw
        @test_logs Extremes.qqplot(fm)

    end

    @testset "returnlevelplot(fm)" begin
        # Plot does not throw
        @test_logs Extremes.returnlevelplot(fm)

    end

    @testset "histplot(fm)" begin
        # Plot does not throw
        @test_logs Extremes.histplot(fm)

    end

    @testset "diagnosticplots(fm)" begin
        # Plots do not throw
        @test_logs Extremes.diagnosticplots(fm)

    end

end

# TODO : Test stationary and non-stationary...
