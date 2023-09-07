@testset "probabilityweightedmoment_gp.jl" begin
  
    df = CSV.read("dataset/gp_stationary.csv", DataFrame)

    deleteat!(df, 101:nrow(df))

    @testset "gpfitpwm(y)" begin
        # stationary model building
        fm = Extremes.gpfitpwm(df.y)

        @test typeof(fm) == pwmAbstractExtremeValueModel{ThresholdExceedance}
        @test fm.model.data.value ≈ df.y
    end

    @testset "gpfitpwm(df, datacol)" begin
        # stationary model building
        fm = Extremes.gpfitpwm(df, :y)

        @test typeof(fm) == pwmAbstractExtremeValueModel{ThresholdExceedance}
        @test fm.model.data.value ≈ df.y
    end

    @testset "gpfitpwm(model)" begin
        # non-stationary warn
        model = Extremes.ThresholdExceedance(Variable("y", df.y), logscalecov = [Variable("t", collect(1:length(df.y)))])

        @test_logs (:warn, "covariates cannot be included in the model when estimating the
            paramters by the probability weighted moment parameter estimation.
            The estimates for the stationary model is returned.") Extremes.gpfitpwm(model)

        # stationary GP fit by pwm
        model = Extremes.ThresholdExceedance(Variable("y", df.y))

        fm = Extremes.gpfitpwm(model)

        @test typeof(fm) == pwmAbstractExtremeValueModel{ThresholdExceedance}
        @test fm.model.data.value ≈ df.y

    end
end
