@testset "maximumlikelihood_gumbel.jl" begin
   
    df = CSV.read("dataset/gev_nonstationary.csv", DataFrame)

    deleteat!(df, 101:nrow(df))

    @testset "gumbelfit(y; locationcov, logscalecov)" begin
        # model building with non-stationary location, logscale and shape
        fm = Extremes.gumbelfit(df.y,
            locationcov = [Variable("x₁", df.x₁)],
            logscalecov = [Variable("x₂", df.x₂)])

        # data is y
        @test all(fm.model.data.value ≈ df.y)

        # location is x₁
        @test length(fm.model.location.covariate) == 1
        @test fm.model.location.covariate[1].value ≈ df.x₁

        # logscale is x₂
        @test length(fm.model.logscale.covariate) == 1
        @test fm.model.logscale.covariate[1].value ≈ df.x₂

        # shape is x₃
        @test length(fm.model.shape.covariate) == 0

    end

    @testset "gumbelfit(y, initialvalues; locationcov, logscalecov)" begin
        # Using initialvalues does not throw nor warn
        @test_logs fm = Extremes.gumbelfit(df.y, zeros(Float64, 4),
            locationcov = [Variable("x₁", df.x₁)],
            logscalecov = [Variable("x₂", df.x₂)])

    end

    @testset "gumbelfit(df, datacol; locationcovid, logscalecovid)" begin
        # model building with non-stationary location and logscale 

        fm = Extremes.gumbelfit(df, :y, locationcovid = [:x₁], logscalecovid = [:x₂])

        # data is y
        @test all(fm.model.data.value .≈ df.y)

        # location is x₁
        @test length(fm.model.location.covariate) == 1
        @test fm.model.location.covariate[1].value ≈ df.x₁

        # logscale is x₂
        @test length(fm.model.logscale.covariate) == 1
        @test fm.model.logscale.covariate[1].value ≈ df.x₂

        # shape is x₃
        @test length(fm.model.shape.covariate) == 0

    end

    @testset "gumbelfit(df, datacol, initialvalues; locationcovid, logscalecovid)" begin
        # Using initialvalues does not throw nor warn

        @test_logs Extremes.gumbelfit(df, :y, zeros(Float64, 4), locationcovid = [:x₁], logscalecovid = [:x₂])

    end

    @testset "gumbelfit(model, initialvalues)" begin
        # non-stationary location and logscale
        model = Extremes.BlockMaxima{Gumbel}(Variable("y", df.y),
            locationcov = [Variable("x₁", df.x₁)],
            logscalecov = [Variable("x₂", df.x₂)])

        fm = Extremes.gumbelfit(model, zeros(Float64, 4))

        # data is y
        @test all(fm.model.data.value .≈ df.y)

        # location is x₁
        @test length(fm.model.location.covariate) == 1
        @test fm.model.location.covariate[1].value ≈ df.x₁

        # logscale is x₂
        @test length(fm.model.logscale.covariate) == 1
        @test fm.model.logscale.covariate[1].value ≈ df.x₂

        # shape is x₃
        @test length(fm.model.shape.covariate) == 0

    end

end
