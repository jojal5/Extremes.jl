"""
# TODO : desc
"""
function probplot(fm::Extremes.MaximumLikelihoodEVA) # TODO : Return value

    y, p̂ = ecdf(fm.model.data)

    dist = Extremes.getdistribution(fm)[]

    p = cdf.(dist, y)

    fig = plot(x=p, y=p̂, Geom.point, Geom.abline(color="red", style=:dash),
        Guide.xlabel("Model"), Guide.ylabel("Empirical"))

end # TODO : Test

"""
# TODO : desc
"""
function qqplot(fm::Extremes.MaximumLikelihoodEVA) # TODO : Return value

    y, p = ecdf(fm.model.data)

    n = length(y)

    q = Vector{Float64}(undef, n)

    for i in eachindex(p)
       q[i] = quantile(fm, p[i])[]
    end

    fig = plot(x=q, y=y, Geom.point, Geom.abline(color="red", style=:dash),
        Guide.xlabel("Model"), Guide.ylabel("Empirical"))

end # TODO : Test

"""
# TODO : desc
"""
function returnlevelplot(fm::Extremes.MaximumLikelihoodEVA) # TODO : Return value

    y, p = ecdf(fm.model.data)

    T = 1 ./ (1 .- p)

    n = length(y)

    q = Vector{Float64}(undef, n)

    for i in eachindex(p)
       q[i] = quantile(fm, p[i])[]
    end

    x = T

    l1 = layer(x=x, y=y, Geom.point)
    l2 = layer(x=x, y=q, Geom.line, Theme(default_color="red", line_style=[:dash]))

    fig = plot(l1,l2, Scale.x_log10, Guide.xlabel("Return Period"), Guide.ylabel("Return Level"))

end # TODO : Test

"""
# TODO : desc
"""
function histplot(fm::Extremes.MaximumLikelihoodEVA) # TODO : Return value

    dist = Extremes.getdistribution(fm)[]

    x = fm.model.data
    n = length(x)
    nbin = Int64(ceil(sqrt(n)))

    xmin = quantile(dist, 1/1000)
    xmax = quantile(dist, 1 - 1/1000)
    xp = range(xmin, xmax, length=1000)


    h = layer(x=x, Geom.histogram(bincount=nbin, density=true))
    d = layer(x=xp, y=pdf.(dist, xp), Geom.line, Theme(default_color="red") )
    plot(d,h, Coord.cartesian(xmin=xmin, xmax=xmax), Guide.xlabel("Data"), Guide.ylabel("Density"))

end # TODO : Test

"""
# TODO : desc
"""
function diagnosticplots(fm::Extremes.MaximumLikelihoodEVA) # TODO : Return value

    f1 = probplot(fm)
    f2 = qqplot(fm)
    f3 = returnlevelplot(fm)
    f4 = histplot(fm)

    return gridstack([f1 f2; f3 f4])
end # TODO : Test
