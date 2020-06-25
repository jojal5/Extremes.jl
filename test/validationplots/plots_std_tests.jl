@testset "plots_std.jl" begin
    n = 5
    y = collect(1:n)
    θ = [0.0, 1.0, 0.1]
    pd = GeneralizedExtremeValue(θ...)
    std = [0.9531, 1.8232, 2.6236, 3.3647, 4.0547]

    fmbm = MaximumLikelihoodEVA(BlockMaxima(Variable("y", y), locationcov = [Variable("t", collect(1:n))]), [0.0, 0.0, 0.0, 0.1])
    fmte = MaximumLikelihoodEVA(ThresholdExceedance(Variable("y", y), logscalecov = [Variable("t", collect(1:n))]), [0.0, 0.0, 0.1])

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

    @testset "standardize(fm)" begin
        # Simple standardized values
        std̂ = Extremes.standardize(fmte)

        @test std ≈ std̂ atol = 0.0001 # TODO : FIX

    end

    @testset "standarddist(fm)" begin
        # BlockMaxima standard distribution is Gumbel
        dist = Extremes.standarddist(fmbm)

        @test dist == Gumbel()

    end

    @testset "standarddist(fm)" begin
        # ThresholdExceedance standard distribution is Exponential
        dist = Extremes.standarddist(fmte)

        @test dist == Exponential()

    end

    @testset "probplot_std_data(fm)" begin
        dfbm = probplot_std_data(fmbm)
        dfte = probplot_std_data(fmte)

        # Returns a dataframe with n values in column Model with BlockMaxima
        @test length(dfbm[:, :Model]) == n

        # Returns a dataframe with n values in column Empirical with BlockMaxima
        @test length(dfbm[:, :Empirical]) == n

        # Returns a dataframe with n values in column Model with ThresholdExceedance
        @test length(dfte[:, :Model]) == n

        # Returns a dataframe with n values in column Empirical with ThresholdExceedance
        @test length(dfte[:, :Empirical]) == n

    end

    @testset "probplot_std(fm)" begin
        # Plot does not throw with BlockMaxima
        @test_logs Extremes.probplot_std(fmbm)

        # Plot does not throw with ThresholdExceedance
        @test_logs Extremes.probplot_std(fmte)

    end

    @testset "qqplot_std_data(fm)" begin
        dfbm = qqplot_std_data(fmbm)
        dfte = qqplot_std_data(fmte)

        # Returns a dataframe with n values in column Model with BlockMaxima
        @test length(dfbm[:, :Model]) == n

        # Returns a dataframe with n values in column Empirical with BlockMaxima
        @test length(dfbm[:, :Empirical]) == n

        # Returns a dataframe with n values in column Model with ThresholdExceedance
        @test length(dfte[:, :Model]) == n

        # Returns a dataframe with n values in column Empirical with ThresholdExceedance
        @test length(dfte[:, :Empirical]) == n

    end

    @testset "qqplot_std(fm)" begin
        # Plot does not throw with BlockMaxima
        @test_logs Extremes.qqplot_std(fmbm)

        # Plot does not throw with ThresholdExceedance
        @test_logs Extremes.qqplot_std(fmte)

    end

    @testset "diagnosticplots_std(fm)" begin
        # Plots do not throw with BlockMaxima
        @test_logs Extremes.diagnosticplots_std(fmbm)

        # Plots do not throw with ThresholdExceedance
        @test_logs Extremes.diagnosticplots_std(fmte)

    end

end
