"""
# TODO : desc
"""
function probplot_data(fm::MaximumLikelihoodEVA)::DataFrame

    checkstationarity(fm)

    y, p̂ = ecdf(fm.model.data.value)

    dist = Extremes.getdistribution(fm)[]

    p = cdf.(dist, y)

    return DataFrame(Model = p, Empirical = p̂)

end

"""
# TODO : desc
"""
function probplot(fm::MaximumLikelihoodEVA)::Plot

    df = probplot_data(fm)

    return plot(df, x=:Model, y=:Empirical, Geom.point, Geom.abline(color="red", style=:dash),
        Guide.xlabel("Model"), Guide.ylabel("Empirical"), Guide.title("Probability Plot"))

end

"""
# TODO : desc
"""
function qqplot_data(fm::MaximumLikelihoodEVA)::DataFrame

    checkstationarity(fm)

    y, p = ecdf(fm.model.data.value)

    n = length(y)

    q = Vector{Float64}(undef, n)

    for i in eachindex(p)
       q[i] = quantile(fm, p[i])[]
    end

    return DataFrame(Model = q, Empirical = y)

end

"""
# TODO : desc
"""
function qqplot(fm::MaximumLikelihoodEVA)::Plot

    df = qqplot_data(fm)

    return plot(df, x=:Model, y=:Empirical, Geom.point, Geom.abline(color="red", style=:dash),
        Guide.xlabel("Model"), Guide.ylabel("Empirical"), Guide.title("Quantile Plot"))

end

"""
# TODO : desc
"""
function returnlevelplot_data(fm::MaximumLikelihoodEVA)::DataFrame

    checkstationarity(fm)

    y, p = ecdf(fm.model.data.value)

    T = 1 ./ (1 .- p)

    n = length(y)

    q = Vector{Float64}(undef, n)

    for i in eachindex(p)
       q[i] = quantile(fm, p[i])[]
    end

    return DataFrame(Data = y, Period = T, Level = q)

end

"""
# TODO : desc
"""
function returnlevelplot(fm::MaximumLikelihoodEVA)::Plot

    df = returnlevelplot_data(fm)

    l1 = layer(df, x=:Period, y=:Level, Geom.point)
    l2 = layer(df, x=:Period, y=:Data, Geom.line, Theme(default_color="red", line_style=[:dash]))

    return plot(l1,l2, Scale.x_log10, Guide.xlabel("Return Period"), Guide.ylabel("Return Level"), Guide.title("Return Level Plot"))

end

"""
# TODO : desc
"""
function histplot_data(fm::MaximumLikelihoodEVA)::Dict

    checkstationarity(fm)

    dist = Extremes.getdistribution(fm)[]

    x = fm.model.data.value
    n = length(x)
    nbin = Int64(ceil(sqrt(n)))

    xmin = quantile(dist, 1/1000)
    xmax = quantile(dist, 1 - 1/1000)
    xp = range(xmin, xmax, length=1000)

    return Dict(:h => DataFrame(Data = x), :d => DataFrame(DataRange = xp, Density = pdf.(dist, xp)),
        :nbin => nbin, :xmin => xmin, :xmax => xmax)

end

"""
# TODO : desc
"""
function histplot(fm::MaximumLikelihoodEVA)::Plot

    dfs = histplot_data(fm)

    h = layer(dfs[:h], x = :Data, Geom.histogram(bincount=dfs[:nbin], density=true))
    d = layer(dfs[:d], x = :DataRange, y = :Density, Geom.line, Theme(default_color="red") )
    return plot(d,h, Coord.cartesian(xmin = dfs[:xmin], xmax = dfs[:xmax]), Guide.xlabel("Data"), Guide.ylabel("Density"), Guide.title("Density Plot"))

end

"""
# TODO : desc
"""
function diagnosticplots(fm::MaximumLikelihoodEVA)::Gadfly.Compose.Context

    f1 = probplot(fm)
    f2 = qqplot(fm)
    f3 = returnlevelplot(fm)
    f4 = histplot(fm)

    return gridstack([f1 f2; f3 f4])
end
