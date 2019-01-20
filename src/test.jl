using CSV, DataFrames, Distributions, Extremes

pd = GeneralizedExtremeValue(0,1,.1)
y = rand(pd,50)
θ̂ = gevfit(y)
θ̂ = Extremes.gevfitlmom(y)
θ̂ = Extremes.gumbelfitpwmom(y)


data = CSV.read("/Users/jalbert/Desktop/montreal.csv")

y = convert(Array{Float64},data[:max])
x = convert(Array{Float64},data[:co2])

θ̂ = gevfit(data[:max])
θ̂ = gevfit(data[:max],location_covariate = x )

θ̂[:params]
