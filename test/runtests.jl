using DataFrames, Dates, SpecialFunctions
using Distributions, Extremes
using Test


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

      data = Dict(:y => y, :n => n)
      dataid = :y
      Covariate = Dict(:μ => Symbol[], :ϕ => Symbol[], :ξ => Symbol[])

      paramindex = Extremes.paramindexing(Covariate, [:μ, :ϕ, :ξ])
      nparameter = 3 + Extremes.getcovariatenumber(Covariate, [:μ, :ϕ, :ξ])

      locationfun = Extremes.computeparamfunction(data, Covariate[:μ])
      logscalefun = Extremes.computeparamfunction(data, Covariate[:ϕ])
      shapefun = Extremes.computeparamfunction(data, Covariate[:ξ])

      model = BlockMaxima(GeneralizedExtremeValue, data, dataid, Covariate,
            locationfun, logscalefun, shapefun, nparameter, paramindex)

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

      data = Dict(:y => y, :x₁ => x₁, :x₂ => x₂, :x₃ => x₃, :n => n)
      dataid = :y
      Covariate = Dict(:μ => [:x₁], :ϕ => [:x₂], :ξ => [:x₃])

      paramindex = Extremes.paramindexing(Covariate, [:μ, :ϕ, :ξ])
      nparameter = 3 + Extremes.getcovariatenumber(Covariate, [:μ, :ϕ, :ξ])

      locationfun = Extremes.computeparamfunction(data, Covariate[:μ])
      logscalefun = Extremes.computeparamfunction(data, Covariate[:ϕ])
      shapefun = Extremes.computeparamfunction(data, Covariate[:ξ])

      model = BlockMaxima(GeneralizedExtremeValue, data, dataid, Covariate,
            locationfun, logscalefun, shapefun, nparameter, paramindex)

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

      data = Dict(:y => y)
      dataid = :y
      Covariate = Dict(:ϕ => Symbol[], :ξ => Symbol[])
      paramindex = Extremes.paramindexing(Covariate, [:ϕ, :ξ])
      nparameter = 2 + Extremes.getcovariatenumber(Covariate, [:ϕ, :ξ])

      model = PeaksOverThreshold(GeneralizedPareto, data, dataid, 1, Covariate, [0], identity, identity, nparameter, paramindex)

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

      data = Dict(:y => y, :n => n, :x₁ => x₁, :x₂ => x₂, :x₃ => x₃)
      dataid = :y
      Covariate = Dict(:ϕ => [:x₁, :x₂], :ξ => [:x₃])
      paramindex = Extremes.paramindexing(Covariate, [:ϕ, :ξ])
      nparameter = 2 + Extremes.getcovariatenumber(Covariate, [:ϕ, :ξ])

      logscalefun = Extremes.computeparamfunction(data, Covariate[:ϕ])
      shapefun = Extremes.computeparamfunction(data, Covariate[:ξ])

      model = PeaksOverThreshold(GeneralizedPareto, data, dataid, 1, Covariate, [0], logscalefun, shapefun, nparameter, paramindex)

      fd = Extremes.getdistribution(model, θ)

      @test fd == pd

end

include("bayesian.jl")
include("maximumlikelihood.jl")
include("probabilityweightedmoment.jl")
include("reproducingColesResults")
include("utils.jl")
