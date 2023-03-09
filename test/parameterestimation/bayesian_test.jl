@testset "bayesian.jl" begin

    # @testset "parametervar(model)" begin

    #     n = 1000

    #     μ = 0.0
    #     σ = 1.0
    #     ξ = 0.1

    #     ϕ = log(σ)
    #     θ = [μ; ϕ; ξ]

    #     pd = GeneralizedExtremeValue(μ, σ, ξ)
    #     y = rand(pd, n)

    #     fm = Extremes.gevfitbayes(y, niter=1000, warmup = 500)
    #     npar = 3 + Extremes.getcovariatenumber(fm.model)
    #     @test size(Extremes.parametervar(fm)) == (npar,npar)

    # end

    @testset "fitbayes(BlockMaxima{GeneralizedExtremeValue}; niter, warmup) -- stationary" begin

        df = CSV.read("dataset/gev_stationary.csv", DataFrame)

        model = Extremes.BlockMaxima{GeneralizedExtremeValue}(Variable("y", df.y))

        fm = Extremes.fitbayes(model, niter=100, warmup=50)

        θ̂ = vec(mean(fm.sim.value, dims=1))

        @test θ̂[1] ≈ 0 atol=0.03
        @test θ̂[2] ≈ 0 atol=0.024
        @test θ̂[3] ≈ 0 atol=0.02

    end

    @testset "fitbayes(BlockMaxima{GeneralizedExtremeValue}; niter, warmup) -- non-stationary" begin

        df = CSV.read("dataset/gev_nonstationary.csv", DataFrame)

        model = Extremes.BlockMaxima{GeneralizedExtremeValue}(Variable("y", df.y),
            locationcov = [Variable("x₁", df.x₁)],
            logscalecov = [Variable("x₂", df.x₂)],
            shapecov = [Variable("x₃", df.x₃)])

        fm = Extremes.fitbayes(model, niter=100, warmup=50)

        θ̂ = vec(mean(fm.sim.value, dims=1))

        @test θ̂[1] ≈ 1 atol=0.03
        @test θ̂[2] ≈ 1 atol=0.03
        @test θ̂[3] ≈ -.5 atol=0.04
        @test θ̂[4] ≈ 1 atol=0.1
        @test θ̂[5] ≈ 0 atol=0.03
        @test θ̂[6] ≈ 0 atol=0.3

    end

    @testset "fitbayes(BlockMaxima{Gumbel}; niter, warmup) -- stationary" begin

        df = CSV.read("dataset/gev_stationary.csv", DataFrame)

        model = Extremes.BlockMaxima{Gumbel}(Variable("y", df.y))

        fm = Extremes.fitbayes(model, niter=100, warmup=50)

        θ̂ = vec(mean(fm.sim.value, dims=1))

        @test θ̂[1] ≈ 0 atol=0.05
        @test θ̂[2] ≈ 0 atol=0.04

    end

    @testset "fitbayes(BlockMaxima{Gumbel}; niter, warmup) -- non-stationary" begin

        df = CSV.read("dataset/gev_nonstationary.csv", DataFrame)

        model = Extremes.BlockMaxima{Gumbel}(Variable("y", df.y),
            locationcov = [Variable("x₁", df.x₁)],
            logscalecov = [Variable("x₂", df.x₂)])

        fm = Extremes.fitbayes(model, niter=100, warmup=50)

        θ̂ = vec(mean(fm.sim.value, dims=1))

        @test θ̂[1] ≈ 1 atol=0.03
        @test θ̂[2] ≈ 1 atol=0.02
        @test θ̂[3] ≈ -.5 atol=0.05
        @test θ̂[4] ≈ 1 atol=0.07

    end

    @testset "fitbayes(ThresholdExceedance) -- stationary" begin

        df = CSV.read("dataset/gp_stationary.csv", DataFrame)

        model = Extremes.ThresholdExceedance(Variable("y", df.y))

        fm = Extremes.fitbayes(model, niter=100, warmup=50)

        θ̂ = vec(mean(fm.sim.value, dims=1))

        @test θ̂[1] ≈ 0 atol=0.07
        @test θ̂[2] ≈ 0 atol=0.05

    end

    @testset "fitbayes(ThresholdExceedance) -- non-stationary" begin

        df = CSV.read("dataset/gp_nonstationary.csv", DataFrame)

        model = Extremes.ThresholdExceedance(Variable("y", df.y),
            logscalecov = [Variable("x₁", df.x₁)])

        fm = Extremes.fitbayes(model, niter=100, warmup=50)

        θ̂ = vec(mean(fm.sim.value, dims=1))

        @test θ̂[1] ≈ -.5 atol=0.09
        @test θ̂[2] ≈ 1 atol=0.15
        @test θ̂[3] ≈ 0 atol=0.05

    end

    include(joinpath("bayesian", "bayesian_gev_test.jl"))
    include(joinpath("bayesian", "bayesian_gp_test.jl"))
    include(joinpath("bayesian", "bayesian_gumbel_test.jl"))

end
