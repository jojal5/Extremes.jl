"""
# TODO : desc
"""
function standardize(fm::Extremes.MaximumLikelihoodEVA) # TODO : Return value

    y = fm.model.data
    d = Extremes.getdistribution(fm)

    μ = location.(d)
    σ = scale.(d)
    ξ = shape.(d)

    z = 1 ./ ξ .* log.( 1 .+ ξ./σ .* ( y .- μ ) )

end # TODO : Test

"""
# TODO : desc
"""
function qqplot_std(fm::Extremes.MaximumLikelihoodEVA) # TODO : Return value

    z = standardize(fm)

    y, p = ecdf(z)

    q = quantile.(Gumbel(0,1), p)

    fig = plot(x=q, y=y, Geom.point, Geom.abline(color="red", style=:dash),
        Guide.xlabel("Model"), Guide.ylabel("Empirical"))

end # TODO : Test

"""
# TODO : desc
"""
function probplot_std(fm::Extremes.MaximumLikelihoodEVA) # TODO : Return value

    z = standardize(fm)

    y, p̂ = ecdf(z)

    dist = Gumbel(0,1)

    p = cdf.(dist, y)

    fig = plot(x=p, y=p̂, Geom.point, Geom.abline(color="red", style=:dash),
        Guide.xlabel("Model"), Guide.ylabel("Empirical"))

end # TODO : Test

"""
# TODO : Desc
"""
function diagnosticplots_std(fm::Extremes.MaximumLikelihoodEVA) # TODO : Return value

    # TODO : Code
end # TODO : Test
