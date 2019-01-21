using CSV, DataFrames, Distributions, Extremes, Gadfly

pd = GeneralizedExtremeValue(0,1,.1)
y = rand(pd,50)
θ̂ = gevfit(y)
θ̂ = Extremes.gevfitlmom(y)
θ̂ = Extremes.gumbelfitpwmom(y)


data = CSV.read("/Users/jalbert/Desktop/gridcell/stat7_69.csv")

y = convert(Array{Float64},data[:max])
x = convert(Array{Float64},data[:co2])

θ̂ = gevfit(data[:max])
θ̂ = gevfit(data[:max],location_covariate = x, initialvalues=[1.22222, 0.000749905, 0.397699, -0.00209455] )

θ̂[:params]

plot(data, x=:co2, y=:max, Geom.point)
