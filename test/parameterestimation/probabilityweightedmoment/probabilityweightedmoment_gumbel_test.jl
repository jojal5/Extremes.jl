@testset "probabilityweightedmoment_gumbel.jl" begin
   
    df = CSV.read("test/dataset/gumbel_stationary.csv", DataFrame)

    @testset "gumbelfitpwm(y)" begin

       fm =  Extremes.gumbelfitpwm(df.y)

       @test typeof(fm.model) == BlockMaxima{Gumbel}

       @test fm.θ̂[1] ≈ -0.0149 atol=0.0001
       @test fm.θ̂[2] ≈ 0.0068 atol=0.0001

    end

    @testset "gumbelfitpwm(df, datacol)" begin
    
        fm = Extremes.gumbelfitpwm(df, :y)

        @test typeof(fm.model) == BlockMaxima{Gumbel}

        @test fm.θ̂[1] ≈ -0.0149 atol=0.0001
        @test fm.θ̂[2] ≈ 0.0068 atol=0.0001

    end

    @testset "gumbelfitpwm(model)" begin
        # non-stationary warn
        model = Extremes.BlockMaxima{Gumbel}(Variable("y", y), locationcov = [Variable("t", collect(1:nrow(df)))])

        @test_logs (:warn, "covariates cannot be included in the model when estimating the
            paramters by the probability weighted moment parameter estimation.
            The estimates for the stationary model is returned.") Extremes.gumbelfitpwm(model)

        # stationary Gumbel fit by pwm
        model = Extremes.BlockMaxima{Gumbel}(Variable("y", df.y))

        fm = Extremes.gumbelfitpwm(model)

        @test typeof(fm.model) == BlockMaxima{Gumbel}

        @test fm.θ̂[1] ≈ -0.0149 atol=0.0001
        @test fm.θ̂[2] ≈ 0.0068 atol=0.0001

    end
end
