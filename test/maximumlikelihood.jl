@testset "GEV fit by ML" begin
      n = 10000

      μ = 0.0
      σ = 1.0
      ξ = 0.1

      ϕ = log(σ)
      θ = [μ; ϕ; ξ]

      pd = GeneralizedExtremeValue(μ, σ, ξ)
      y = rand(pd, n)

      fm = Extremes.gevfit(y)

      @test fm.θ̂ ≈ θ atol = 0.05

end


@testset "non-stationary location GEV fit by ML" begin
      n = 10000

      x₁ = randn(n)
      x₂ = randn(n)

      μ = x₁ + x₂
      σ = 1.0
      ξ = 0.1

      ϕ = log(σ)
      θ = [0.0; 1.0; 1.0; ϕ; ξ]

      pd = GeneralizedExtremeValue.(μ, σ, ξ)
      y = rand.(pd)

      fm = Extremes.gevfit(y; locationcov = [x₁, x₂])

      @test fm.θ̂ ≈ θ atol = 0.05

end

@testset "non-stationary logscale GEV fit by ML" begin
      n = 10000

      x₁ = randn(n) / 3
      x₂ = randn(n) / 3

      μ = 0.0
      ϕ = x₁ + x₂
      ξ = 0.1

      σ = exp.(ϕ)
      θ = [μ; 0.0; 1.0; 1.0; ξ]

      pd = GeneralizedExtremeValue.(μ, σ, ξ)
      y = rand.(pd)

      fm = Extremes.gevfit(y, scalecov = [x₁, x₂])

      @test fm.θ̂ ≈ θ atol = 0.05

end

@testset "non-stationary location and logscale GEV fit by ML" begin
      n = 10000

      x₁ = randn(n)
      x₂ = randn(n) / 3

      μ = 1.0 .+ x₁
      ϕ = -.05 .+ x₂
      ξ = 0.1

      σ = exp.(ϕ)
      θ = [1.0; 1.0; -.05; 1.0; ξ]

      pd = GeneralizedExtremeValue.(μ, σ, ξ)
      y = rand.(pd)

      fm = Extremes.gevfit(y, locationcov = [x₁], scalecov = [x₂])

      @test fm.θ̂ ≈ θ atol = 0.05

end

@testset "non-stationary shape GEV fit by ML" begin
      n = 10000

      x₁ = randn(n) / 10

      μ = 0.0
      ϕ = 0.0
      ξ = x₁

      σ = exp(ϕ)
      θ = [μ; ϕ; 0.0; 1.0]

      pd = GeneralizedExtremeValue.(μ, σ, ξ)
      y = rand.(pd)

      fm = Extremes.gevfit(y, shapecov = [x₁])

      @test fm.θ̂ ≈ θ atol = 0.1

end

################################################################################
# Generalized Pareto
################################################################################

@testset "GP fit by ML" begin
      n = 10000

      σ = 1.0
      ξ = 0.1

      ϕ = log(σ)
      θ = [ϕ; ξ]

      pd = GeneralizedPareto(σ, ξ)
      y = rand(pd, n)

      fm = Extremes.gpfit(y, n) # TODO : n is probability not nobservation

      @test fm.θ̂ ≈ θ atol = 0.05

end

@testset "non-stationary logscale GP fit by ML" begin
      n = 10000

      x₁ = randn(n) / 3
      x₂ = randn(n) / 3

      ϕ = -.5 .+ x₁ .+ x₂
      ξ = 0.1

      σ = exp.(ϕ)
      θ = [-.5; 1.0; 1.0; ξ]

      pd = GeneralizedPareto.(σ, ξ)
      y = rand.(pd)

      data = Dict(:y => y, :x₁ => x₁, :x₂ => x₂, :n => n)
      dataid = :y
      Covariate = Dict(:ϕ => [:x₁, :x₂])

      fm = Extremes.gpfit(data, dataid, n, Covariate = Covariate) # TODO : n is probability not nobservation

      @test fm.θ̂ ≈ θ atol = 0.05

end


@testset "non-stationary shape GP fit by ML" begin
      n = 10000

      x₁ = randn(n) / 10

      μ = 0.0
      ϕ = 0.0
      ξ = x₁

      σ = exp(ϕ)
      θ = [ϕ; 0.0; 1.0]

      pd = GeneralizedPareto.(μ, σ, ξ)
      y = rand.(pd)

      data = Dict(:y => y, :x₁ => x₁, :n => n)
      dataid = :y
      Covariate = Dict(:ξ => [:x₁])

      fm = Extremes.gpfit(data, dataid, n, Covariate = Covariate)

      @test fm.θ̂ ≈ θ atol = 0.1

end
