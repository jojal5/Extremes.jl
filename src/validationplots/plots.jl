"""
# TODO : desc
"""
function probplot(fm::MaximumLikelihoodEVA)::Plot

    y, p̂ = ecdf(fm.model.data.value)

    dist = Extremes.getdistribution(fm)[]

    p = cdf.(dist, y)

    return plot(x=p, y=p̂, Geom.point, Geom.abline(color="red", style=:dash),
        Guide.xlabel("Model"), Guide.ylabel("Empirical"))

end

"""
# TODO : desc
"""
function qqplot(fm::MaximumLikelihoodEVA)::Plot

    y, p = ecdf(fm.model.data.value)

    n = length(y)

    q = Vector{Float64}(undef, n)

    for i in eachindex(p)
       q[i] = quantile(fm, p[i])[]
    end

    return plot(x=q, y=y, Geom.point, Geom.abline(color="red", style=:dash),
        Guide.xlabel("Model"), Guide.ylabel("Empirical"))

end

"""
# TODO : desc
"""
function returnlevelplot(fm::MaximumLikelihoodEVA)::Plot

    y, p = ecdf(fm.model.data.value)

    T = 1 ./ (1 .- p)

    n = length(y)

    q = Vector{Float64}(undef, n)

    for i in eachindex(p)
       q[i] = quantile(fm, p[i])[]
    end

    x = T

    l1 = layer(x=x, y=y, Geom.point)
    l2 = layer(x=x, y=q, Geom.line, Theme(default_color="red", line_style=[:dash]))

    return plot(l1,l2, Scale.x_log10, Guide.xlabel("Return Period"), Guide.ylabel("Return Level"))

end

"""
# TODO : desc
"""
function histplot(fm::MaximumLikelihoodEVA)::Plot

    dist = Extremes.getdistribution(fm)[]

    x = fm.model.data.value
    n = length(x)
    nbin = Int64(ceil(sqrt(n)))

    xmin = quantile(dist, 1/1000)
    xmax = quantile(dist, 1 - 1/1000)
    xp = range(xmin, xmax, length=1000)


    h = layer(x=x, Geom.histogram(bincount=nbin, density=true))
    d = layer(x=xp, y=pdf.(dist, xp), Geom.line, Theme(default_color="red") )
    return plot(d,h, Coord.cartesian(xmin=xmin, xmax=xmax), Guide.xlabel("Data"), Guide.ylabel("Density"))

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
