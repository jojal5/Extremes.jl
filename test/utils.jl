
@testset "Slicing matrix" begin
      m = 10
      n = 5
      A = rand(1:10, m, n)

      B = Extremes.slicematrix(A)
      C = Extremes.slicematrix(A, dims = 2)

      Aᴮ = Extremes.unslicematrix(B)
      Aᶜ = Extremes.unslicematrix(C, dims = 2)

      @test length(B) == n
      @test length(C) == m
      @test A == Aᴮ
      @test A == Aᶜ
end;


@testset "Probability weighted moment with p=1, r=0 and s=k" begin
      μ = .5
      σ = 1.0
      pd = Logistic(μ,σ)
      y = rand(pd, 1000)
      # theoritical form of the pwm for the Logistic distribution (Greenwood and al., 1979)
      f(k::Int) = μ/(1+k) - σ/(1+k)*sum(1/i for i=1:k)

      for k=1:3
            @test f(k) ≈ Extremes.pwm(y,1,0,k) rtol=.1
      end
end;

@testset "Probability weighted moment with p=1, r=j and s=0" begin
      μ = .5
      σ = 1.0
      ξ = 0
      pd = GeneralizedExtremeValue(μ,σ,0)
      y = rand(pd, 1000)

      # theoritical form of the pwm for the Gumbel distribution (Greenwood and al., 1979)
      f(j::Int) = μ/(1+j) + σ/(1+j) * (log(1+j) + MathConstants.eulergamma)

      for j=1:3
            @test f(j) ≈ Extremes.pwm(y,1,j,0) rtol=.1
      end
end;
