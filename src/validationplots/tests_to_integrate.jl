using Extremes, DataFrames, Gadfly, Test, Distributions

data = load("portpirie")

x = Variable("x", randn(100))
μ = 10 .+ x.value
σ = 1.0
ξ = .1
pd = GeneralizedExtremeValue.(μ, σ, ξ)
y = rand.(pd)


fm = gevfit(data, :SeaLevel)
fmns = gevfit(y, locationcov=[x])

probplot(fm)
probplot(fmns)

qqplot(fm)
qqplot(fmns)

returnlevelplot(fm)
returnlevelplot(fmns)

histplot(fm)
histplot(fmns)

diagnosticplots(fm)
diagnosticplots(fmns)

fm = gevfitbayes(data, :SeaLevel)
fmns = gevfitbayes(y, locationcov=[x])

probplot(fm)
probplot(fmns)

qqplot(fm)
qqplot(fmns)

returnlevelplot(fm)
returnlevelplot(fmns)

histplot(fm)
histplot(fmns)

diagnosticplots(fm)
diagnosticplots(fmns)



ys = rand(GeneralizedPareto(1,.1),100)

x = Variable("x", randn(100)/3)
ϕ = x.value
σ = exp.(ϕ)
ξ = .1
pd = GeneralizedPareto.(σ, ξ)
y = rand.(pd)


fm = gpfit(ys)
fmns = gpfit(y, logscalecov=[x])

probplot(fm)
probplot(fmns)

qqplot(fm)
qqplot(fmns)

returnlevelplot(fm)
returnlevelplot(fmns)

histplot(fm)
histplot(fmns)

diagnosticplots(fm)
diagnosticplots(fmns)

fm = gpfitbayes(ys)
fmns = gpfitbayes(y, logscalecov=[x])

probplot(fm)
probplot(fmns)

qqplot(fm)
qqplot(fmns)

returnlevelplot(fm)
returnlevelplot(fmns)

histplot(fm)
histplot(fmns)

diagnosticplots(fm)
diagnosticplots(fmns)
