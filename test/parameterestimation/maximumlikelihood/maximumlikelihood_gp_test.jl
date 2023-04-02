@testset "maximumlikelihood_gp.jl" begin
    
    # df = CSV.read("dataset/gp_nonstationary.csv", DataFrame)
    df = CSV.read("test/dataset/gp_nonstationary.csv", DataFrame)

    deleteat!(df, 101:nrow(df))

    @testset "gpfit(y; logscalecov, shapecov)" begin
        # model building with non-stationary logscale and shape

        x₁ = Variable("x₁", df.x₁)
        y = df.y

        fm = Extremes.gpfit(y,
            logscalecov = [x₁],
            shapecov = [x₁])

        # data is y
        @test fm.model.data.value ≈ y

        # logscale is x₁
        @test length(fm.model.logscale.covariate) == 1
        @test fm.model.logscale.covariate[1].value ≈ df.x₁

        # shape is x₂
        @test length(fm.model.shape.covariate) == 1
        @test fm.model.shape.covariate[1].value ≈ df.x₁

    end

    @testset "gpfit(y, initialvalues; logscalecov, shapecov)" begin
        # model building with non-stationary logscale and shape

        x₁ = Variable("x₁", df.x₁)
        y = df.y

        @test_logs Extremes.gpfit(y, zeros(Float64, 4),
            logscalecov = [x₁],
            shapecov = [x₁])
    end

    @testset "gpfit(df, datacol; logscalecovid, shapecovid)" begin
        # model building with non-stationary location, logscale and shape
        
        fm = Extremes.gpfit(df, :y, logscalecovid = [:x₁], shapecovid = [:x₁])

        # data is y
        @test fm.model.data.value ≈ df.y

        # logscale is x₁
        @test length(fm.model.logscale.covariate) == 1
        @test fm.model.logscale.covariate[1].value ≈ df.x₁

        # shape is x₂
        @test length(fm.model.shape.covariate) == 1
        @test fm.model.shape.covariate[1].value ≈ df.x₁

    end

    @testset "gpfit(df, datacol, initialvalues; logscalecovid, shapecovid)" begin
        # model building with non-stationary location, logscale and shape

        @test_logs Extremes.gpfit(df, :y, zeros(Float64, 4), logscalecovid = [:x₁], shapecovid = [:x₁])

    end

    @testset "gpfit(model, initialvalues)" begin
        # non-stationary location, logscale and shape

        x₁ = Variable("x₁", df.x₁)

        model = Extremes.ThresholdExceedance(Variable("y", df.y),
            logscalecov = [x₁],
            shapecov = [x₁])

        fm = Extremes.gpfit(model, zeros(Float64, 4))

        # data is y
        @test fm.model.data.value ≈ df.y

        # logscale is x₁
        @test length(fm.model.logscale.covariate) == 1
        @test fm.model.logscale.covariate[1].value ≈ df.x₁

        # shape is x₂
        @test length(fm.model.shape.covariate) == 1
        @test fm.model.shape.covariate[1].value ≈ df.x₁

    end

end
