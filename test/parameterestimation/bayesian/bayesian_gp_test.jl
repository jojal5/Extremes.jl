@testset "bayesian_gp.jl" begin
    
    df = CSV.read("dataset/gp_nonstationary.csv", DataFrame)

    deleteat!(df, 101:nrow(df))

    y = df.y
    x₁ = Variable("x₁", df.x₁)

    @testset "gpfitbayes(y; logscalecov, shapecov, niter, warmup)" begin
        # model building with non-stationary logscale and shape
        fm = Extremes.gpfitbayes(y,
            logscalecov = [x₁],
            shapecov = [x₁],
            niter=2, warmup=1)

            # data is y
            @test fm.model.data.value ≈ y

            # logscale is x₁
            @test length(fm.model.logscale.covariate) == 1
            @test fm.model.logscale.covariate[1].value ≈ x₁.value

            # shape is x₂
            @test length(fm.model.shape.covariate) == 1
            @test fm.model.shape.covariate[1].value ≈ x₁.value

    end

    @testset "gpfitbayes(df, datacol; logscalecovid, shapecovid, niter, warmup)" begin
        # model building with non-stationary location, logscale and shape
        
        fm = Extremes.gpfitbayes(df, :y, 
            logscalecovid = [:x₁], 
            shapecovid = [:x₁], 
            niter=2, warmup=1)

        # data is y
        @test fm.model.data.value ≈ y

        # logscale is x₁
        @test length(fm.model.logscale.covariate) == 1
        @test fm.model.logscale.covariate[1].value ≈ x₁.value

        # shape is x₂
        @test length(fm.model.shape.covariate) == 1
        @test fm.model.shape.covariate[1].value ≈ x₁.value

    end

    @testset "gpfitbayes(model; niter, warmup)" begin
        # non-stationary location, logscale and shape
        model = Extremes.ThresholdExceedance(Variable("y", y),
            logscalecov = [x₁],
            shapecov = [x₁])

        fm = Extremes.gpfitbayes(model, niter=2, warmup=1)

        # data is y
        @test fm.model.data.value ≈ y

        # logscale is x₁
        @test length(fm.model.logscale.covariate) == 1
        @test fm.model.logscale.covariate[1].value ≈ x₁.value

        # shape is x₂
        @test length(fm.model.shape.covariate) == 1
        @test fm.model.shape.covariate[1].value ≈ x₁.value

    end

end
