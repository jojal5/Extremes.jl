using CSV, DataFrames
using Distributions, Extremes

pd = GeneralizedExtremeValue(0,1,.1)
y = rand(pd,50)

pdfit = Extremes.gumbelfitpwmom(y)
pdfit = Extremes.gevfitlmom(y)

ini = Extremes.getinitialvalues(y)

gevfit(y)

x = collect(0:1/100:2)
pd = GeneralizedExtremeValue.(x,1,.1)
y = rand.(pd)

# data = CSV.read("/Users/jalbert/Desktop/stat7_69.csv")
# y = convert(Array{Float64},data[:max])
# x = convert(Array{Float64},data[:co2])

ini = Extremes.getinitialvalues(y)
splice!(ini,2:1,0)

fd = gevfit(y, x)
fd = gevfit(y, x, initialvalues = ini)



pd = GeneralizedPareto(0,1,.1)
y = rand(pd,500)

fd = Extremes.gpdfitmom(y)
fd = Extremes.gpdfitmom(y, threshold = 0)

insupport(fd,y)

fd = Extremes.gpdfit(y)
fd = Extremes.gpdfit(y, threshold = 0)

pd = GeneralizedPareto(10,1,.1)
y = rand(pd,500)

fd = Extremes.gpdfitmom(y, threshold = 10)
fd = Extremes.gpdfit(y, threshold = 10)
