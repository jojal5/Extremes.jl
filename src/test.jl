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

fd = Extremes.gevfitbayes(y)
fd = Extremes.gevfitbayes(y, stepSize=[.025,.05,.08])
fd = Extremes.gevfitbayes(y,niter=5000, stepSize=[.025,.05,.08])
fd = Extremes.gevfitbayes(y,warmup=1000, niter=5000, stepSize=[.025,.05,.08])
fd = Extremes.gevfitbayes(y,warmup=1000, thin=4, niter=5000, stepSize=[.025,.05,.08])
fd = Extremes.gevfitbayes(y, warmup=10000, niter=5000, stepSize=[.025,.05,.08])


y = rand(Normal(),100)
c = Extremes.getcluster(y,.2)
c = Extremes.getcluster(y,.2,.1)

threshold = 5
σ = 1
ξ = .1
y = rand(GeneralizedPareto(threshold, σ, ξ),100)

Extremes.gpdfitbayes(y)
Extremes.gpdfitbayes(y, threshold=threshold)
Extremes.gpdfitbayes(y, threshold=threshold, stepSize=[.2,.15])
