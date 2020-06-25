"""
# TODO : desc
"""
function standardize(y::Real, μ::Real, σ::Real, ξ::Real)::Real
    return 1 / ξ * log( 1 + ξ/σ * ( y - μ ) )

end

"""
# TODO : desc
"""
function standardize(fm::MaximumLikelihoodEVA{BlockMaxima{T}} where T<:Distribution)::Vector{<:Real}

    y = fm.model.data.value
    d = Extremes.getdistribution(fm)

    return standardize.(y, location.(d), scale.(d), shape.(d))

end

"""
# TODO : desc
"""
function standardize(fm::MaximumLikelihoodEVA{ThresholdExceedance})::Vector{<:Real}

    y = fm.model.data.value
    d = Extremes.getdistribution(fm)

    return standardize.(y, 0, scale.(d), shape.(d))

end

"""
# TODO : desc
"""
function standarddist(fm::MaximumLikelihoodEVA{BlockMaxima{T}} where T<:Distribution)::Distribution
    return Gumbel()
end

"""
# TODO : desc
"""
function standarddist(fm::MaximumLikelihoodEVA{ThresholdExceedance})::Distribution
    return Exponential()
end

"""
# TODO : desc
"""
function probplot_std_data(fm::MaximumLikelihoodEVA)::DataFrame

    checknonstationarity(fm)

    z = standardize(fm)

    y, p̂ = ecdf(z)

    return DataFrame(Model = cdf.(standarddist(fm), y), Empirical = p̂)

end

"""
# TODO : desc
"""
function probplot_std(fm::MaximumLikelihoodEVA)::Plot

    df = probplot_std_data(fm)

    return plot(df, x=:Model, y=:Empirical, Geom.point, Geom.abline(color="red", style=:dash),
        Guide.xlabel("Model"), Guide.ylabel("Empirical"), Guide.title("Residual Probability Plot"))

end

"""
# TODO : desc
"""
function qqplot_std_data(fm::MaximumLikelihoodEVA)::DataFrame

    checknonstationarity(fm)

    z = standardize(fm)

    y, p = ecdf(z)

    return DataFrame(Model = quantile.(standarddist(fm), p), Empirical = y)

end

"""
# TODO : desc
"""
function qqplot_std(fm::MaximumLikelihoodEVA)::Plot

    df = qqplot_std_data(fm)

    return plot(df, x=:Model, y=:Empirical, Geom.point, Geom.abline(color="red", style=:dash),
        Guide.xlabel("Model"), Guide.ylabel("Empirical"), Guide.title("Residual Quantile Plot"))

end

"""
# TODO : Desc
"""
function diagnosticplots_std(fm::MaximumLikelihoodEVA)::Gadfly.Compose.Context
    probplot = probplot_std(fm)
    qqplot = qqplot_std(fm)

    return hstack(probplot, qqplot)

end
