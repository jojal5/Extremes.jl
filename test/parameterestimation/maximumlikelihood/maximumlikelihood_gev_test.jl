@testset "maximumlikelihood_gev.jl" begin
   
    df = CSV.read("dataset/gev_nonstationary.csv", DataFrame)

    deleteat!(df, 101:nrow(df))

    @testset "gevfit(y; locationcov, logscalecov, shapecov)" begin
        # model building with non-stationary location, logscale and shape
        fm = Extremes.gevfit(df.y,
            locationcov = [Variable("x₁", df.x₁)],
            logscalecov = [Variable("x₂", df.x₂)],
            shapecov = [Variable("x₃", df.x₃)])

        # data is y
        @test all(fm.model.data.value ≈ df.y)

        # location is x₁
        @test length(fm.model.location.covariate) == 1
        @test fm.model.location.covariate[1].value ≈ df.x₁

        # logscale is x₂
        @test length(fm.model.logscale.covariate) == 1
        @test fm.model.logscale.covariate[1].value ≈ df.x₂

        # shape is x₃
        @test length(fm.model.shape.covariate) == 1
        @test fm.model.shape.covariate[1].value ≈ df.x₃

    end

    @testset "gevfit(y, initialvalues; locationcov, logscalecov, shapecov)" begin
        # Using initialvalues does not throw nor warn
        @test_logs fm = Extremes.gevfit(df.y, zeros(Float64, 6),
            locationcov = [Variable("x₁", df.x₁)],
            logscalecov = [Variable("x₂", df.x₂)],
            shapecov = [Variable("x₃", df.x₃)])

    end

    @testset "gevfit(df, datacol; locationcovid, logscalecovid, shapecovid)" begin
        # model building with non-stationary location, logscale and shape

        fm = Extremes.gevfit(df, :y, locationcovid = [:x₁], logscalecovid = [:x₂], shapecovid = [:x₃])

        # data is y
        @test all(fm.model.data.value .≈ df.y)

        # location is x₁
        @test length(fm.model.location.covariate) == 1
        @test fm.model.location.covariate[1].value ≈ df.x₁

        # logscale is x₂
        @test length(fm.model.logscale.covariate) == 1
        @test fm.model.logscale.covariate[1].value ≈ df.x₂

        # shape is x₃
        @test length(fm.model.shape.covariate) == 1
        @test fm.model.shape.covariate[1].value ≈ df.x₃

    end

    @testset "gevfit(df, datacol, initialvalues; locationcovid, logscalecovid, shapecovid)" begin
        # Using initialvalues does not throw nor warn

        @test_logs Extremes.gevfit(df, :y, zeros(Float64, 6), locationcovid = [:x₁], logscalecovid = [:x₂], shapecovid = [:x₃])

    end

    @testset "gevfit(model, initialvalues)" begin
        # non-stationary location, logscale and shape
        model = Extremes.BlockMaxima{GeneralizedExtremeValue}(Variable("y", df.y),
            locationcov = [Variable("x₁", df.x₁)],
            logscalecov = [Variable("x₂", df.x₂)],
            shapecov = [Variable("x₃", df.x₃)])

        fm = Extremes.gevfit(model, zeros(Float64, 6))

        # data is y
        @test all(fm.model.data.value .≈ df.y)

        # location is x₁
        @test length(fm.model.location.covariate) == 1
        @test fm.model.location.covariate[1].value ≈ df.x₁

        # logscale is x₂
        @test length(fm.model.logscale.covariate) == 1
        @test fm.model.logscale.covariate[1].value ≈ df.x₂

        # shape is x₃
        @test length(fm.model.shape.covariate) == 1
        @test fm.model.shape.covariate[1].value ≈ df.x₃

    end

end
