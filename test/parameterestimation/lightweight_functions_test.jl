@testset "lightweight_functions.m"

    @testset "fit(GeneralizedExtremeValue, y).jl" begin

        df = CSV.read("dataset/gev_stationary.csv", DataFrame)
        y = df.y
            
        # Assert estimation method
        @test_throws AssertionError fit(GeneralizedExtremeValue, z, method = "unsupported")
            
        fm_mle = fit_mle(GeneralizedExtremeValue, y)   

        @test all(isapprox.(params(fm_mle),(0.0009, exp(0.0142), -0.0060), atol=.0001))
            
        fm_pwm = fit_pwm(GeneralizedExtremeValue, y) 
        
        @test all(isapprox.(params(fm_pwm),(-0.0005, exp(0.0125), -0.0033), atol=.0001))
    
    end

    @testset "fit(Gumbel, y).jl" begin

        df = CSV.read("dataset/gev_stationary.csv", DataFrame)
        y = df.y
            
        # Assert estimation method
        @test_throws AssertionError fit(Gumbel, y, method = "unsupported")
            
        fm_mle = fit_mle(Gumbel, y)   
    
        @test all(isapprox.(params(fm_mle),(-0.0023, exp(0.0124)), atol=.0001))
            
        fm_pwm = fit_pwm(Gumbel, y) 
        
        @test all(isapprox.(params(fm_pwm),(-0.0020, exp(0.0095)), atol=.0001))
      
    end

    @testset "fit(GeneralizedPareto, y).jl" begin

        df = CSV.read("dataset/gp_stationary.csv", DataFrame)
        y = df.y
            
        # Assert estimation method
        @test_throws AssertionError fit(GeneralizedPareto, y, method = "unsupported")
            
        fm_mle = fit_mle(GeneralizedPareto, y)   
    
        @test all(isapprox.(params(fm_mle), (0., exp(-0.0135), 0.0059), atol=.0001))
            
        fm_pwm = fit_pwm(GeneralizedPareto, y) 
        
        @test all(isapprox.(params(fm_pwm),(0., exp(-0.0199), 0.0122), atol=.0001))
      
    end

end