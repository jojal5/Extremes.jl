df = load("rain")

threshold = 30

data = filter(row -> row.Rainfall > threshold, df)

y = data[:, :Rainfall] .- threshold

fm = gpfit(y, threshold = [threshold])

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
