using DataFrames, Dates, SpecialFunctions
using Distributions, Extremes
using Test
using LinearAlgebra

@testset "Test of getdistribution" begin

      # Model BlockMaxima - stationary
      n = 100

      μ = 0.0
      σ = 1.0
      ξ = 0.1
      ϕ = log(σ)

      θ = [μ; ϕ; ξ]

      pd = GeneralizedExtremeValue(μ, σ, ξ)
      y = rand(pd, n)

      model = BlockMaxima(y)

      fd = Extremes.getdistribution(model, θ)[]

      @test fd == pd

      # BlockMaxima model - non-stationary

      n = 10000

      x₁ = randn(n)
      x₂ = randn(n)/3
      x₃ = randn(n)/10

      θ = [5.0 ; 1.0 ; -.5 ; 1.0 ; 0.0 ; 1.0]

      μ = θ[1] .+ θ[2] * x₁
      ϕ = θ[3] .+ θ[4] * x₂
      ξ = θ[5] .+ θ[6] * x₃

      pd = GeneralizedExtremeValue.(μ, exp.(ϕ), ξ)

      y = rand.(pd)

      model = BlockMaxima(y, locationcov = [x₁], scalecov = [x₂], shapecov = [x₃])

      fd = Extremes.getdistribution(model, θ)

      @test pd == fd


      # PeaksOverThreshold model - stationary

      n = 100

      σ = 1.0
      ξ = .1
      ϕ = log(σ)

      θ = [ϕ ; ξ]

      pd = GeneralizedPareto(σ, ξ)
      y = rand(pd,n)

      model = PeaksOverThreshold(y, n) # TODO : n is probability not nobservation

      fd = Extremes.getdistribution(model, θ)[]

      @test fd == pd

      # PeaksOverThreshold model - non-stationary

      n = 100

      x₁ = randn(n)/3
      x₂ = randn(n)/3
      x₃ = randn(n)/10

      θ = [-.5 ; 1.0 ; 1.0 ; 0 ; 1.0]

      ϕ = θ[1] .+ θ[2] * x₁ .+ θ[3] * x₂
      ξ = θ[4] .+ θ[5] * x₃

      σ = exp.(ϕ)

      pd = GeneralizedPareto.(σ, ξ)
      y = rand.(pd)

      model = PeaksOverThreshold(y, n, scalecov = [x₁, x₂], shapecov = [x₃]) # TODO : n is probably not nobservation

      fd = Extremes.getdistribution(model, θ)

      @test fd == pd

end

include("bayesian.jl")
include("maximumlikelihood.jl")
include("probabilityweightedmoment.jl")
include("reproducingColesResults.jl")
include("utils.jl")
