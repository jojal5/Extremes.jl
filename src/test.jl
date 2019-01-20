using Distributions, Extremes
pd = GeneralizedExtremeValue(0,1,.1)
y = rand(pd,50)
θ̂ = gevfit(y)


x = collect(0:.05:3)
μ = 0 .+ 1*x

σ = 1

ξ = .1

pd = GeneralizedExtremeValue.(μ,σ,ξ)
y = rand.(pd)
θ̂ =  gevfit(y, location_covariate=x)
