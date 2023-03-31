@testset "bayesian_gumbel.jl" begin
    
    df = CSV.read("dataset/gev_nonstationary.csv", DataFrame)

    deleteat!(df, 101:nrow(df))

    @testset "gumbelfitbayes(y; locationcov, logscalecov, niter, warmup)" begin
        # model building with non-stationary location and logscale
        fm = Extremes.gumbelfitbayes(df.y,
            locationcov = [Variable("x₁", df.x₁)],
            logscalecov = [Variable("x₂", df.x₂)],
            niter=10, warmup=5)

        # data is y
        @test fm.model.data.value ≈ df.y

        # location is x₁
        @test length(fm.model.location.covariate) == 1
        @test fm.model.location.covariate[1].value ≈ df.x₁

        # logscale is x₂
        @test length(fm.model.logscale.covariate) == 1
        @test fm.model.logscale.covariate[1].value ≈ df.x₂

        # shape is x₃
        @test length(fm.model.shape.covariate) == 0

    end

    @testset "gumbelfitbayes(df, datacol; locationcovid, logscalecovid, niter, warmup)" begin
        # model building with non-stationary location and logscale

        fm = Extremes.gumbelfitbayes(df, :y, locationcovid = [:x₁], logscalecovid = [:x₂], niter=10, warmup=5)

        # data is y
        @test fm.model.data.value ≈ df.y

        # location is x₁
        @test length(fm.model.location.covariate) == 1
        @test fm.model.location.covariate[1].value ≈ df.x₁

        # logscale is x₂
        @test length(fm.model.logscale.covariate) == 1
        @test fm.model.logscale.covariate[1].value ≈ df.x₂

        # shape is x₃
        @test length(fm.model.shape.covariate) == 0

    end

    @testset "gumbelfitbayes(model; niter, warmup)" begin
        # non-stationary location and logscale
        model = Extremes.BlockMaxima{Gumbel}(Variable("y", df.y),
            locationcov = [Variable("x₁", df.x₁)],
            logscalecov = [Variable("x₂", df.x₂)])

        fm = Extremes.gumbelfitbayes(model, niter=10, warmup=5)

        # data is y
        @test fm.model.data.value ≈ df.y

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
