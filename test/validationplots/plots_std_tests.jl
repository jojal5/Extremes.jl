@testset "plots_std.jl" begin
    n = 5
    y = collect(1:n)
    pd = GeneralizedExtremeValue(1.0, 1.0, 0.1)
    std = [0, 0.9531, 1.8232, 2.6236, 3.3647]

    fm = MaximumLikelihoodEVA(BlockMaxima(Variable("y", y)), [1.0, 0.0, 0.1])

    @testset "standardize(fm)" begin
        # Simple standardized values
        std̂ = Extremes.standardize(fm)

        @test std ≈ std̂ atol = 0.0001

    end

    @testset "probplot_std_data(fm)" begin
        df = probplot_std_data(fm)

        # Returns a dataframe with n values in column Model
        @test length(df[:, :Model]) == n

        # Returns a dataframe with n values in column Empirical
        @test length(df[:, :Empirical]) == n

    end

    @testset "probplot_std(fm)" begin
        # Plot does not throw
        @test_logs Extremes.probplot_std(fm)

    end

    @testset "qqplot_std_data(fm)" begin
        df = qqplot_std_data(fm)

        # Returns a dataframe with n values in column Model
        @test length(df[:, :Model]) == n

        # Returns a dataframe with n values in column Empirical
        @test length(df[:, :Empirical]) == n

    end

    @testset "qqplot_std(fm)" begin
        # Plot does not throw
        @test_logs Extremes.qqplot_std(fm)

    end

    @testset "diagnosticplots_std(fm)" begin
        # Plots do not throw
        @test_logs Extremes.diagnosticplots_std(fm)

    end

end
