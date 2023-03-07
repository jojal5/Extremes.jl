@testset "probabilityweightedmoment_gev.jl" begin

    df = CSV.read("test/dataset/gev_stationary.csv", DataFrame)

    @testset "gevfitpwm(y)" begin

        fm = Extremes.gevfitpwm(df.y)

        @test typeof(fm.model) == BlockMaxima{GeneralizedExtremeValue}

        @test fm.θ̂[1] ≈ -0.0005 atol=0.0001
        @test fm.θ̂[2] ≈ 0.0125 atol=0.0001
        @test fm.θ̂[3] ≈ -0.0033 atol=0.0001

    end

    @testset "gevfitpwm(df, datacol)" begin

        fm = Extremes.gevfitpwm(df, :y)

        @test typeof(fm.model) == BlockMaxima{GeneralizedExtremeValue}

        @test fm.θ̂[1] ≈ -0.0005 atol=0.0001
        @test fm.θ̂[2] ≈ 0.0125 atol=0.0001
        @test fm.θ̂[3] ≈ -0.0033 atol=0.0001

    end

    @testset "gevfitpwm(model)" begin
        # non-stationary warn
        model = Extremes.BlockMaxima{GeneralizedExtremeValue}(Variable("y", y), locationcov = [Variable("t", collect(1:nrow(df)))])

        @test_logs (:warn, "covariates cannot be included in the model when estimating the
            paramters by the probability weighted moment parameter estimation.
            The estimates for the stationary model is returned.") Extremes.gevfitpwm(model)

        # stationary GEV fit by pwm
        model = Extremes.BlockMaxima{GeneralizedExtremeValue}(Variable("y", df.y))

        fm = Extremes.gevfitpwm(model)

        @test fm.θ̂[1] ≈ -0.0005 atol=0.0001
        @test fm.θ̂[2] ≈ 0.0125 atol=0.0001
        @test fm.θ̂[3] ≈ -0.0033 atol=0.0001

    end
end
