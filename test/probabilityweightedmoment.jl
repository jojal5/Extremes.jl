@testset "GEV fit by pwm" begin
      n = 10000
      θ = [0.0;1.0;.2]

      pd = GeneralizedExtremeValue(θ...)
      y = rand(pd, n)

      fm = Extremes.gevfitpwm(y)

      @test fm.θ̂ ≈ θ rtol = .05

end;

@testset "GP fit by pwm" begin
      n = 10000
      θ = [1.0 ; .2]

      pd = GeneralizedPareto(θ...)
      y = rand(pd, n)

      fm = Extremes.gpfitpwm(y, n) # TODO : n is probability not nobservation

      @test fm.θ̂ ≈ θ rtol = .05

end;

@testset "Gumbel fit by pwm" begin
      n = 10000
      θ = [0.0 ; 1.0]

      pd = Gumbel(θ...)
      y = rand(pd, n)

      fm = Extremes.gumbelfitpwm(y)

      @test fm.θ̂[1:2] ≈ θ rtol = .05

end;
