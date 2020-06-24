@testset "plots_std.jl" begin
    n = 5
    y = collect(1:n)
    threshold = 1.0
    θ = [threshold, 1.0, 0.1]
    pd = GeneralizedExtremeValue(θ...)
    std = [0, 0.9531, 1.8232, 2.6236, 3.3647]

    fmbm = MaximumLikelihoodEVA(BlockMaxima(Variable("y", y)), [θ[1], log(θ[2]), θ[3]])
    fmte = MaximumLikelihoodEVA(ThresholdExceedance(Variable("y", y)), [log(θ[2]), θ[3]])

    @testset "standardize(y, μ, σ, ξ)" begin
        # Simple standardized values
        std̂ = Extremes.standardize(y[2], θ...)

        @test std[2] ≈ std̂ atol = 0.0001

    end

    @testset "standardize(fm)" begin
        # Simple standardized values
        std̂ = Extremes.standardize(fmbm)

        @test std ≈ std̂ atol = 0.0001

    end

    @testset "standardize(fm, threshold)" begin
        # Simple standardized values
        std̂ = Extremes.standardize(fmte, threshold)

        @test std ≈ std̂ atol = 0.0001

    end

    @testset "probplot_std_data(fm)" begin
        df = probplot_std_data(fmbm)

        # Returns a dataframe with n values in column Model
        @test length(df[:, :Model]) == n

        # Returns a dataframe with n values in column Empirical
        @test length(df[:, :Empirical]) == n

    end

    @testset "probplot_std_data(fm, threshold)" begin
        df = probplot_std_data(fmte, threshold)

        # Returns a dataframe with n values in column Model
        @test length(df[:, :Model]) == n

        # Returns a dataframe with n values in column Empirical
        @test length(df[:, :Empirical]) == n

    end

    @testset "probplot_std(df)" begin
        # Plot does not throw
        @test_logs Extremes.probplot_std(DataFrame(Model = collect(1:n), Empirical = collect(1:n)))

    end

    @testset "probplot_std(fm)" begin
        # Plot does not throw
        @test_logs Extremes.probplot_std(fmbm)

    end

    @testset "probplot_std(fm, threshold)" begin
        # Plot does not throw
        @test_logs Extremes.probplot_std(fmte, threshold)

    end

    @testset "qqplot_std_data(fm)" begin
        df = qqplot_std_data(fmbm)

        # Returns a dataframe with n values in column Model
        @test length(df[:, :Model]) == n

        # Returns a dataframe with n values in column Empirical
        @test length(df[:, :Empirical]) == n

    end

    @testset "qqplot_std_data(fm, threshold)" begin
        df = qqplot_std_data(fmte, threshold)

        # Returns a dataframe with n values in column Model
        @test length(df[:, :Model]) == n

        # Returns a dataframe with n values in column Empirical
        @test length(df[:, :Empirical]) == n

    end

    @testset "qqplot_std(df)" begin
        # Plot does not throw
        @test_logs Extremes.qqplot_std(DataFrame(Model = collect(1:n), Empirical = collect(1:n)))

    end

    @testset "qqplot_std(fm)" begin
        # Plot does not throw
        @test_logs Extremes.qqplot_std(fmbm)

    end

    @testset "qqplot_std(fm, threshold)" begin
        # Plot does not throw
        @test_logs Extremes.qqplot_std(fmte, threshold)

    end

    @testset "diagnosticplots_std(fm)" begin
        # Plots do not throw
        @test_logs Extremes.diagnosticplots_std(fmbm)

    end

    @testset "diagnosticplots_std(fm, threshold)" begin
        # Plots do not throw
        @test_logs Extremes.diagnosticplots_std(fmte, threshold)

    end

end
