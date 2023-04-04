@testset "bayesian_gev.jl" begin

    df = CSV.read("dataset/gev_nonstationary.csv", DataFrame)

    deleteat!(df, 101:nrow(df))

    y = df.y
    x₁ = Variable("x₁", df.x₁)
    x₂ = Variable("x₂", df.x₂)
    x₃ = Variable("x₃", df.x₃)

    @testset "gevfitbayes(y; locationcov, logscalecov, shapecov, niter, warmup)" begin
        # model building with non-stationary location, logscale and shape
        fm = Extremes.gevfitbayes(y,
            locationcov = [x₁],
            logscalecov = [x₂],
            shapecov = [x₃],
            niter=2, warmup=1)

        # data is y
        @test fm.model.data.value ≈ y

        # location is x₁
        @test length(fm.model.location.covariate) == 1
        @test fm.model.location.covariate[1].value ≈ x₁.value

        # logscale is x₂
        @test length(fm.model.logscale.covariate) == 1
        @test fm.model.logscale.covariate[1].value ≈ x₂.value

        # shape is x₃
        @test length(fm.model.shape.covariate) == 1
        @test fm.model.shape.covariate[1].value ≈ x₃.value

    end

    @testset "gevfitbayes(df, datacol; locationcovid, logscalecovid, shapecovid, niter, warmup)" begin
        # model building with non-stationary location, logscale and shape
        
        fm = Extremes.gevfitbayes(df, :y, 
            locationcovid = [:x₁], 
            logscalecovid = [:x₂], 
            shapecovid = [:x₃], 
            niter=2, warmup=1)

        # data is y
        @test fm.model.data.value ≈ y

        # location is x₁
        @test length(fm.model.location.covariate) == 1
        @test fm.model.location.covariate[1].value ≈ x₁.value

        # logscale is x₂
        @test length(fm.model.logscale.covariate) == 1
        @test fm.model.logscale.covariate[1].value ≈ x₂.value

        # shape is x₃
        @test length(fm.model.shape.covariate) == 1
        @test fm.model.shape.covariate[1].value ≈ x₃.value

    end

    @testset "gevfitbayes(model; niter, warmup)" begin
        # non-stationary location, logscale and shape
        model = Extremes.BlockMaxima{GeneralizedExtremeValue}(Variable("y", y),
            locationcov = [x₁],
            logscalecov = [x₂],
            shapecov = [x₃])

        fm = Extremes.gevfitbayes(model, niter=2, warmup=1)

        # data is y
        @test fm.model.data.value ≈ y

        # location is x₁
        @test length(fm.model.location.covariate) == 1
        @test fm.model.location.covariate[1].value ≈ x₁.value

        # logscale is x₂
        @test length(fm.model.logscale.covariate) == 1
        @test fm.model.logscale.covariate[1].value ≈ x₂.value

        # shape is x₃
        @test length(fm.model.shape.covariate) == 1
        @test fm.model.shape.covariate[1].value ≈ x₃.value

    end

end
