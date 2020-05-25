using DataFrames, Dates
using Distributions, Extremes
using Test

pd = GeneralizedExtremeValue(0, 1, 0.1)
y = rand(pd, 50)

# Testing the utility functions
pdfit = Extremes.gumbelfitpwmom(y)
pdfit = Extremes.gevfitlmom(y)
Extremes.getinitialvalue(GeneralizedExtremeValue, y)

A = rand(5, 10)
B = Extremes.slicematrix(A)
Ã = Extremes.unslicematrix(B)

B = Extremes.slicematrix(A, dims = 2)
Ã = Extremes.unslicematrix(B)


# Testing stationary GEV fit with maximum likelihood

fd = gevfit(y)
Extremes.getdistribution(fd)
Extremes.getdistribution(fd.model, [0, 0, 0])
Extremes.quantile(fd.model, [0, 0, 0], 0.95)
Extremes.quantile(fd, 0.95)
Extremes.parametervar(fd)
Extremes.quantilevar(fd, 0.95)

data = Dict(:y => y)
dataid = :y
Covariate = Dict(:μ => Symbol[], :ϕ => Symbol[], :ξ => Symbol[])
paramindex = Extremes.paramindexing(Covariate, [:μ, :ϕ, :ξ])
nparameter = Extremes.getparameternumber(Covariate)
model = Extremes.BlockMaxima(
    GeneralizedExtremeValue,
    data,
    dataid,
    Covariate,
    identity,
    identity,
    identity,
    nparameter,
    paramindex,
)
Extremes.getdistribution(model, [0, 0, 0])

fd = gevfit(model)

# Testing non-stationary GEV fit (only location) with maximum likelihood
n = 300
x = collect(1:n)
μ = x * 1 / 100
pd = GeneralizedExtremeValue.(μ, 1, 0.1)
y = rand.(pd)
data = Dict(:y => y, :x => x, :n => n)
Covariate = Dict(:μ => [:x], :ϕ => Symbol[], :ξ => Symbol[])
fd = gevfit(data, :y, Covariate = Covariate)
Extremes.getdistribution(fd)
Extremes.getdistribution(fd.model, [0, 0, 0, 0])
Extremes.quantile(fd.model, [0, 0, 0, 0], 0.95)
Extremes.quantile(fd, 0.95)
Extremes.parametervar(fd)
Extremes.quantilevar(fd, 0.95)

model = Extremes.BlockMaxima(
    GeneralizedExtremeValue,
    data,
    dataid,
    Covariate,
    identity,
    identity,
    identity,
    nparameter,
    paramindex,
)
fittedmodel = gevfit(model)


# Testing non-stationary GEV fit (location and scale) with maximum likelihood
n = 300
x = collect(1:n)
μ = x * 1 / 100
ϕ = x * 1 / 1000
pd = GeneralizedExtremeValue.(μ, exp.(ϕ), 0.1)
y = rand.(pd)
data = Dict(:y => y, :x => x, :n => n)
Covariate = Dict(:μ => [:x], :ϕ => [:x], :ξ => Symbol[])

fd = gevfit(data, :y, Covariate = Covariate)
Extremes.getdistribution(fd)
Extremes.getdistribution(fd.model, [0, 0, 0, 0, 0])
Extremes.quantile(fd.model, [0, 0, 0, 0, 0], 0.95)
Extremes.quantile(fd, 0.95)
Extremes.parametervar(fd)
Extremes.quantilevar(fd, 0.95)




# Testing stationary GEV fit with non-informative Bayesian
pd = GeneralizedExtremeValue(0, 1, 0.1)
y = rand(pd, 300)
fm = gevfitbayes(y)
Extremes.quantile(fm, 0.95)

# Testing non-stationary GEV (only location) fit (only location) with maximum likelihood
n = 300
x = collect(1:n)
μ = x * 1 / 100
pd = GeneralizedExtremeValue.(μ, 1, 0.1)
y = rand.(pd)
data = Dict(:y => y, :x => x, :n => n)
Covariate = Dict(:μ => [:x], :ϕ => Symbol[], :ξ => Symbol[])
fm = gevfitbayes(data, :y, Covariate = Covariate, niter = 1000, warmup = 500)
Extremes.quantile(fm, 0.95)


pd = GeneralizedPareto(0,1,.1)
y = rand(pd, 300)

fm = Extremes.gpfit(y)
fm = Extremes.gpfit(y, threshold = [0], nobsperblock = 1)
Extremes.getdistribution(fm)
Extremes.getdistribution(fm.model, [0, 0])
Extremes.quantile(fm.model, [0, 0], 0.95)
Extremes.quantile(fm, 0.95)
Extremes.parametervar(fm)
Extremes.quantilevar(fm, 0.95)

data = Dict(:y => y)
dataid = :y
Covariate = Dict(:μ => Symbol[], :ϕ => Symbol[], :ξ => Symbol[])

fm = gpfit(data, :y)

fm = Extremes.gpfit(fm.model)




println("End of tests")
# fd = gevfit(data, dataid=:y)
# fd = gevfit(data, dataid=:y, initialvalues=[0, 0, .1])
# fd = gevfit(data, dataid=:y, locationid=:x)
# fd = gevfit(data, dataid=:y, locationid=:x, initialvalues=[0, 1/50, 0, .1])
# fd = gevfit(data, dataid=:y, logscaleid=:x)
# fd = gevfit(data, dataid=:y, logscaleid=:x, initialvalues=[0, 0, 0, .1])
# fd = gevfit(data, dataid=:y, locationid=:x, logscaleid=:x)
# fd = gevfit(data, dataid=:y, locationid=:x, logscaleid=:x, initialvalues=[0,1/50,0,0,.1])
#
# @test_throws AssertionError gevfit(data, dataid=:z)
# @test_throws AssertionError gevfit(data, dataid=:y, locationid=:z)
# @test_throws AssertionError gevfit(data, dataid=:y, logscaleid=:z)
#
# # test for many datasets in a loop
# for i=1:1000
#     x = collect(0:300)
#     μ = x/100
#     pd = GeneralizedExtremeValue.(μ,1,.1)
#     y = rand.(pd)
#     data = Dict(:y => y, :x => x)
#     gevfit(data, dataid=:y, locationid=:x, logscaleid=:x)
# end
#
#
#
# # Test for the GPD
#
# pd = Normal()
# y = rand(pd,1000)
# g = getcluster(y,1,0)
#
# x = collect(Date(2000,1,1):Day(1):Date(2010,12,31))
# y = rand(pd, length(x))
# df = DataFrame(Date = x, Y = y)
# g = getcluster(df,1,0)
#
# df = DataFrame(Y = y, Date = x)
# @test_throws AssertionError getcluster(df,1,0)
# df = DataFrame(Date = x, Y=x)
# @test_throws AssertionError getcluster(df,1,0)

# x = collect(0:300)
# σ = exp.(x/300)
# pd = GeneralizedPareto.(0,σ,.1)
# y = rand.(pd)
# data = Dict(:y => y, :x => x)
#
# fd = Extremes.gpdfitmom(y)
# fd = Extremes.gpdfitmom(y, threshold = -0.05)
#
# ini = Extremes.getinitialvalue(GeneralizedPareto,y)
#
# fd = gpdfit(y)
# fd = gpdfit(y, threshold=-.5)
# fd = gpdfit(y, threshold=-.5, initialvalues=[1, .1])
# fd = gpdfit(data, dataid=:y)
# fd = gpdfit(data, dataid=:y, threshold=-.05)
# fd = gpdfit(data, dataid=:y, initialvalues=[1, .1])
# fd = gpdfit(data, dataid=:y, logscaleid=:x)
# fd = gpdfit(data, dataid=:y, logscaleid=:x, initialvalues=[1, 0, .1])
#
# @test_throws AssertionError gpdfit(data, dataid=:z)
# @test_throws AssertionError gpdfit(data, dataid=:y, logscaleid=:z)
#
# # test for many datasets in a loop
# for i=1:1000
#     x = collect(0:300)
#     σ = exp.(x/300)
#     pd = GeneralizedPareto.(0,σ,.1)
#     y = rand.(pd)
#     data = Dict(:y => y, :x => x)
#     gpdfit(data, dataid=:y, logscaleid=:x)
# end
#
#
#
# fd = Extremes.gevfitbayes(y)
# fd = Extremes.gevfitbayes(y, stepSize=[.025,.05,.08])
# fd = Extremes.gevfitbayes(y,niter=5000, stepSize=[.025,.05,.08])
# fd = Extremes.gevfitbayes(y,warmup=1000, niter=5000, stepSize=[.025,.05,.08])
# fd = Extremes.gevfitbayes(y,warmup=1000, thin=4, niter=5000, stepSize=[.025,.05,.08])
# fd = Extremes.gevfitbayes(y, warmup=2000, niter=5000, stepSize=[.025,.05,.08])
#
#
# y = rand(Normal(),100)
# c = Extremes.getcluster(y,.2)
# c = Extremes.getcluster(y,.2,.1)
#
# threshold = 5
# σ = 1
# ξ = .1
# y = rand(GeneralizedPareto(threshold, σ, ξ),100)
#
# Extremes.gpdfitbayes(y)
# Extremes.gpdfitbayes(y, threshold=threshold)
# Extremes.gpdfitbayes(y, threshold=threshold, stepSize=[.2,.15])
