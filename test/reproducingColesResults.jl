@testset "Reproducing Coles Results" begin
    @testset "Port Pirie example from Coles (2001)" begin

        data = load("portpirie")

        # Fit of the GEV distribution by maximum likelihood
        fm = gevfit(data, :SeaLevel)

        # Parameter estimates
        θ̂ = fm.θ̂

        # Approximate variance-covariance matrix of the parameter estimates
        V̂ = Extremes.parametervar(fm)

        # correction factor for using ϕ instead of σ (computation using the delta method)
        c = [
            1.0 exp(θ̂[2]) 1.0
            exp(θ̂[2]) exp(2 * θ̂[2]) exp(θ̂[2])
            1.0 exp(θ̂[2]) 1.0
        ]

        # 10-year return level
        R₁₀ = returnlevel(fm, 10, 0.95)
        # 100-year return level
        R₁₀₀ = returnlevel(fm, 100, 0.95)

        # Parameter estimates in Coles
        θ = [3.87; log(0.198); -0.050]

        # Approximate variance-covariance matrix of the parameter estimates in Coles
        V = [
            0.000780 0.000197 -.00107
            0.000197 0.000410 -.000778
            -.00107 -.000778 0.00965
        ]

        @test θ̂ ≈ θ rtol = 0.1
        @test c .* V̂ ≈ V rtol = 0.1
        @test Extremes.loglike(fm.model, θ̂) ≈ 4.34 rtol = 0.1
        @test R₁₀.value[] ≈ 4.30 rtol = 0.1
        @test R₁₀.cint[] ≈ [4.19; 4.45] rtol = 0.1
        @test R₁₀₀.value[] ≈ 4.69 rtol = 0.1
        @test R₁₀₀.cint[] ≈ [4.5; 5.27] rtol = 0.1

    end

    @testset "Fremantle example from Coles (2001)" begin

        df = load("fremantle")

        n = length(df[:, :SeaLevel])

        t = collect(1:n)

        fm = gevfit(df[:, :SeaLevel], locationcov = [ExplanatoryVariable("t", t)])

        # Parameter estimates
        θ̂ = fm.θ̂

        # Variance of the parameter estimates
        V̂ = diag(Extremes.parametervar(fm))

        # Computing the variance of σ instead of ϕ with the delta method
        V̂[3] = exp(2 * θ̂[3]) * V̂[3]

        # Computing the standard errors of the parameters estimates
        Ŝ = sqrt.(V̂)

        # Values given in Coles (2001, Chap 6) for the temporal linear trend in μ
        θ = [1.38; 0.00203; log(0.124); -.125]
        S = [0.03; 0.00052; 0.010; 0.070]


        @test θ̂ ≈ θ rtol = 0.1
        @test Ŝ ≈ S rtol = 0.1
        @test Extremes.loglike(fm.model, θ̂) ≈ 49.9 rtol = 0.1

        evt = ExplanatoryVariable("t", t)
        evt² = ExplanatoryVariable("t²", t .^ 2)
        evSOI = ExplanatoryVariable("SOI", df[:, :SOI])

        # Quadratic trend in μ
        fm = gevfit(df[:, :SeaLevel], locationcov = [evt, evt²])
        @test Extremes.loglike(fm.model, fm.θ̂) ≈ 50.6 rtol = 0.1

        # Linear trend in both μ and ϕ
        fm = gevfit(df[:, :SeaLevel], locationcov = [evt], logscalecov = [evt])
        @test Extremes.loglike(fm.model, fm.θ̂) ≈ 50.7 rtol = 0.1

        # μ as a function of the SOI
        fm = gevfit(df, :SeaLevel, locationcovid = [:SOI])
        @test Extremes.loglike(fm.model, fm.θ̂) ≈ 47.2 rtol = 0.1

        # Linear trend in μ as function of the time and the SOI
        fm = gevfit(df[:, :SeaLevel], locationcov = [evt, evSOI])
        @test Extremes.loglike(fm.model, fm.θ̂) ≈ 53.9 rtol = 0.1
        @test fm.θ̂[3] ≈ 0.055 rtol = 0.1

    end

    @testset "Rainfall example from Coles (2001)" begin
        df = load("rain")

        threshold = 30

        data = filter(row -> row.Rainfall > threshold, df)

        y = data[:, :Rainfall] .- threshold

        fm = gpfit(y)

        θ̂ = fm.θ̂

        # Approximate variance-covariance matrix of the parameter estimates
        V̂ = Extremes.parametervar(fm)

        # correction factor for using ϕ instead of σ (computation using the delta method)
        c = [
            exp(2*θ̂[1]) exp(θ̂[1])
            exp(θ̂[1]) 1.0
        ]

        V̂ = c.*V̂
        V = [.9188 -.0655 ;
            -.0655 .0102]

        @test fm.θ̂ ≈ [log(7.44); 0.184] rtol = 0.1
        @test Extremes.loglike(fm.model, fm.θ̂) ≈ -485.1 rtol = 0.1
        @test V̂ ≈ V rtol = 0.1

        r = returnlevel(fm, [threshold], size(df,1), 365, 100)

        @test r.value[] ≈ 106.3 rtol = .1

    end

    @testset "Dow Jones example from Coles (2001), Section 4.4.2" begin
        df = load("dowjones")

        X = df[:,:Index]


        X̃ = 100*[ log(X[i+1]) - log(X[i])  for i in 1:(length(X)-1)]

        n = length(X̃)

        threshold = 2.0

        z = X̃[X̃ .> threshold] .- threshold

        fm = gpfit(z)

        θ̂ = fm.θ̂

        # Approximate variance-covariance matrix of the parameter estimates
        V̂ = Extremes.parametervar(fm)

        # correction factor for using ϕ instead of σ (computation using the delta method)
        c = [
            exp(2*θ̂[1]) exp(θ̂[1])
            exp(θ̂[1]) 1.0
        ]

        V̂ = c.*V̂

        # Test the Generalized Pareto estimates
        @test exp(fm.θ̂[1]) ≈ .495 rtol = 0.1
        @test fm.θ̂[2] ≈ .288 rtol = 0.1

        # Test the standard errors of estimates
        @test sqrt(V̂[1,1]) ≈ .150 rtol = 0.1
        @test sqrt(V̂[2,2]) ≈ .258 rtol = 0.1

    end
end
