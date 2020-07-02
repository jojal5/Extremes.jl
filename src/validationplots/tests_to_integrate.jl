using Extremes, DataFrames, Gadfly, Test, Distributions

import Extremes.probplot_std_data, Extremes.qqplot_std_data

"""
# TODO : desc
"""
function probplot(fm::fittedEVA)::Plot

    if Extremes.getcovariatenumber(fm.model) > 0
        df = probplot_std_data(fm)
        plotTitle = "Residual Probability Plot"
    else
        df = probplot_data(fm)
        plotTitle = "Probability Plot"
    end

    return plot(df, x=:Model, y=:Empirical, Geom.point, Geom.abline(color="red", style=:dash),
        Guide.xlabel("Model"), Guide.ylabel("Empirical"), Guide.title(plotTitle))

end

"""
# TODO : desc
"""
function qqplot(fm::fittedEVA)::Plot

    if Extremes.getcovariatenumber(fm.model) > 0
        df = qqplot_std_data(fm)
        plotTitle = "Residual Quantile Plot"
    else
        df = qqplot_data(fm)
        plotTitle = "Quantile Plot"
    end

    return plot(df, x=:Model, y=:Empirical, Geom.point, Geom.abline(color="red", style=:dash),
        Guide.xlabel("Model"), Guide.ylabel("Empirical"), Guide.title(plotTitle))

end

"""
# TODO : desc
"""
function returnlevelplot(fm::fittedEVA)::Plot

    if Extremes.getcovariatenumber(fm.model) > 0
        @warn "this graphic is not optimized for non-stationary model; plot ignored."
        return plot()
    else
        df = returnlevelplot_data(fm)
        l1 = layer(df, x=:Period, y=:Level,Geom.line, Theme(default_color="red", line_style=[:dash]))
        l2 = layer(df, x=:Period, y=:Data, Geom.point)
        return plot(l1,l2, Scale.x_log10, Guide.xlabel("Return Period"), Guide.ylabel("Return Level"), Guide.title("Return Level Plot"))
    end
end

"""
# TODO : desc
"""
function histplot(fm::fittedEVA)::Plot

    if Extremes.getcovariatenumber(fm.model) > 0
        df = histplot_std_data(fm)
        plotTitle = "Residual Density Plot"
    else
        df = histplot_data(fm)
        plotTitle = "Density Plot"
    end

    h = layer(df[:h], x = :Data, Geom.histogram(bincount=df[:nbin], density=true))
    d = layer(df[:d], x = :DataRange, y = :Density, Geom.line, Theme(default_color="red") )

    return plot(d,h, Coord.cartesian(xmin = df[:xmin], xmax = df[:xmax]), Guide.xlabel("Data"), Guide.ylabel("Density"), Guide.title(plotTitle))

end

"""
# TODO : desc
"""
function diagnosticplots(fm::fittedEVA)::Gadfly.Compose.Context

    f1 = probplot(fm)
    f2 = qqplot(fm)
    f3 = histplot(fm)

    if Extremes.getcovariatenumber(fm.model) > 0
        f4 = plot()
    else
        f4 = histplot(fm)
    end

    return gridstack([f1 f2; f3 f4])
end

"""
# TODO : desc
"""
function probplot_std_data(fm::fittedEVA)::DataFrame

    z = Extremes.standardize(fm)

    y, p̂ = Extremes.ecdf(z)

    return DataFrame(Model = cdf.(Extremes.standarddist(fm.model), y), Empirical = p̂)

end


"""
# TODO : desc
"""
function qqplot_std_data(fm::fittedEVA)::DataFrame

    z = Extremes.standardize(fm)

    y, p = Extremes.ecdf(z)

    return DataFrame(Model = quantile.(Extremes.standarddist(fm.model), p), Empirical = y)

end


"""
# TODO : desc
"""
function histplot_std_data(fm::fittedEVA)::Dict

    dist = Extremes.standarddist(fm.model)

    z = Extremes.standardize(fm)
    n = length(z)
    nbin = Int64(ceil(sqrt(n)))

    zmin = quantile(dist, 1/1000)
    zmax = quantile(dist, 1 - 1/1000)
    zp = range(zmin, zmax, length=1000)

    return Dict(:h => DataFrame(Data = z), :d => DataFrame(DataRange = zp, Density = pdf.(dist, zp)),
        :nbin => nbin, :xmin => zmin, :xmax => zmax)

end




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
