@testset "plots_std.jl" begin
    y = collect(1:5)
    pd = GeneralizedExtremeValue(1.0, 1.0, 0.1)
    std = [0, 0.9531, 1.8232, 2.6236, 3.3647]

    fm = MaximumLikelihoodEVA(BlockMaxima(Variable("y", y)), [1.0, 0.0, 0.1])

    @testset "standardize(fm)" begin
        # Simple standardized values
        std̂ = Extremes.standardize(fm)

        @test std ≈ std̂ atol = 0.0001

    end

    @testset "qqplot_std(fm)" begin
        # Plot does not throw
        @test_logs Extremes.qqplot_std(fm)

    end

    @testset "probplot_std(fm)" begin
        # Plot does not throw
        @test_logs Extremes.probplot_std(fm)

    end

    @testset "diagnosticplots_std(fm)" begin
        # Plots do not throw
        @test_logs Extremes.diagnosticplots_std(fm)

    end

end
