@testset "plots.jl" begin
    n = 5000

    y = rand(GeneralizedExtremeValue(1.0, 1.0, 0.1), n)
    fm = MaximumLikelihoodEVA(BlockMaxima(Variable("y", y)), [1.0, 0.0, 0.1])

    x = rand(n)
    yns = rand.(GeneralizedExtremeValue.(x, 1.0, 0.1))
    fmns = MaximumLikelihoodEVA(BlockMaxima(Variable("y", yns), locationcov = [Variable("x", x)]), [0.0, 1.0, 0.0, 0.1])

    @testset "probplot_data(fm)" begin
        df = probplot_data(fm)

        # Returns a dataframe with n values in column Model
        @test length(df[:, :Model]) == n

        # Returns a dataframe with n values in column Empirical
        @test length(df[:, :Empirical]) == n

        # Info for non-stationary model but does not throw
        @test_logs (:info, "The graph is optimized for stationary models and the model provided is not.") probplot_data(fmns)

    end

    @testset "probplot(fm)" begin
        # Plot does not throw
        @test_logs probplot(fm)

    end

    @testset "qqplot_data(fm)" begin
        df = qqplot_data(fm)

        # Returns a dataframe with n values in column Model
        @test length(df[:, :Model]) == n

        # Returns a dataframe with n values in column Empirical
        @test length(df[:, :Empirical]) == n

        # Info for non-stationary model but does not throw
        @test_logs (:info, "The graph is optimized for stationary models and the model provided is not.") qqplot_data(fmns)

    end

    @testset " qqplot(fm)" begin
        # Plot does not throw
        @test_logs qqplot(fm)

    end

    @testset "returnlevelplot_data(fm)" begin
        df = returnlevelplot_data(fm)

        # Returns a dataframe with n values in column Data
        @test length(df[:, :Data]) == n

        # Returns a dataframe with n values in column Period
        @test length(df[:, :Period]) == n

        # Returns a dataframe with n values in column Level
        @test length(df[:, :Level]) == n

        # Info for non-stationary model but does not throw
        @test_logs (:info, "The graph is optimized for stationary models and the model provided is not.") returnlevelplot_data(fmns)

    end

    @testset "returnlevelplot(fm)" begin
        # Plot does not throw
        @test_logs returnlevelplot(fm)

    end

    @testset "histplot_data(fm)" begin
        dfs = histplot_data(fm)

        # Returns a dataframe with n values in column Data
        @test length(dfs[:h][:, :Data]) == n

        # Returns a dataframe with 1000 values in column DataRange
        @test length(dfs[:d][:, :DataRange]) == 1000

        # Returns a dataframe with 1000 values in column Density
        @test length(dfs[:d][:, :Density]) == 1000

        # Returns a dictionary with key nbin
        @test haskey(dfs, :nbin)

        # Returns a dictionary with key xmin
        @test haskey(dfs, :xmin)

        # Returns a dictionary with key xmax
        @test haskey(dfs, :xmax)

        # Info for non-stationary model but does not throw
        @test_logs (:info, "The graph is optimized for stationary models and the model provided is not.") histplot_data(fmns)

    end

    @testset "histplot(fm)" begin
        # Plot does not throw
        @test_logs histplot(fm)

    end

    @testset "diagnosticplots(fm)" begin
        # Plots do not throw
        @test_logs diagnosticplots(fm)

    end

    @testset "mrl(y)" begin

        σ = 1.0
        ξ = 0.1
        pd = GeneralizedPareto(σ, ξ)
        y = rand(pd, 1000)

        df = mrl(y, 10)

        # Returns a dictionary with the correct keys
        @test haskey(df, :Threshold)
        @test haskey(df, :mrl)
        @test haskey(df, :lbound)
        @test haskey(df, :ubound)

        # Theoretical values
        m = (σ .+ ξ * df[:, :Threshold]) / (1 - ξ)

        @test all(df[:, :lbound] .< m .< df[:, :ubound])

    end

end
