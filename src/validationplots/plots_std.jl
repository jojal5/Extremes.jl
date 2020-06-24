"""
# TODO : desc
"""
function standardize(fm::MaximumLikelihoodEVA{BlockMaxima{T}} where T<:Distribution)::Vector{<:Real}

    y = fm.model.data.value
    d = Extremes.getdistribution(fm)

    μ = location.(d)
    σ = scale.(d)
    ξ = shape.(d)

    return 1 ./ ξ .* log.( 1 .+ ξ./σ .* ( y .- μ ) )

end

"""
# TODO : desc
"""
function probplot_std(fm::MaximumLikelihoodEVA{BlockMaxima{T}} where T<:Distribution)::Plot

    z = standardize(fm)

    y, p̂ = ecdf(z)

    dist = Gumbel(0,1)

    p = cdf.(dist, y)

    return plot(x=p, y=p̂, Geom.point, Geom.abline(color="red", style=:dash),
        Guide.xlabel("Model"), Guide.ylabel("Empirical"), Guide.title("Probability Plot"), Guide.title("Residual Probability Plot"))

end

"""
# TODO : desc
"""
function qqplot_std(fm::MaximumLikelihoodEVA{BlockMaxima{T}} where T<:Distribution)::Plot

    z = standardize(fm)

    y, p = ecdf(z)

    q = quantile.(Gumbel(0,1), p)

    return plot(x=q, y=y, Geom.point, Geom.abline(color="red", style=:dash),
        Guide.xlabel("Model"), Guide.ylabel("Empirical"), Guide.title("Residual Quantile Plot"))

end

"""
# TODO : Desc
"""
function diagnosticplots_std(fm::MaximumLikelihoodEVA{BlockMaxima{T}} where T<:Distribution)::Gadfly.Compose.Context
    probplot = probplot_std(fm)
    qqplot = qqplot_std(fm)

    return hstack(probplot, qqplot)

end
