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
function standardize(fm::MaximumLikelihoodEVA{ThresholdExceedance}, threshold::Real)::Vector{<:Real}

    y = fm.model.data.value
    d = Extremes.getdistribution(fm)

    return standardize.(y, threshold, scale.(d), shape.(d))

end

"""
# TODO : desc
"""
function probplot_std_data(fm::MaximumLikelihoodEVA{BlockMaxima{T}} where T<:Distribution)::DataFrame

    z = standardize(fm)

    y, p̂ = ecdf(z)

    return DataFrame(Model = cdf.(Gumbel(), y), Empirical = p̂)

end

"""
# TODO : desc
"""
function probplot_std_data(fm::MaximumLikelihoodEVA{ThresholdExceedance}, threshold::Real)::DataFrame

    z = standardize(fm, threshold)

    y, p̂ = ecdf(z)

    return DataFrame(Model = cdf.(Exponential(), y), Empirical = p̂)

end

"""
# TODO : desc
"""
function probplot_std(df::DataFrame)::Plot

    return plot(df, x=:Model, y=:Empirical, Geom.point, Geom.abline(color="red", style=:dash),
        Guide.xlabel("Model"), Guide.ylabel("Empirical"), Guide.title("Residual Probability Plot"))

end

"""
# TODO : desc
"""
function probplot_std(fm::MaximumLikelihoodEVA{BlockMaxima{T}} where T<:Distribution)::Plot

    df = probplot_std_data(fm)

    return probplot_std(df)

end

"""
# TODO : desc
"""
function probplot_std(fm::MaximumLikelihoodEVA{ThresholdExceedance}, threshold::Real)::Plot

    df = probplot_std_data(fm, threshold)

    return probplot_std(df)

end

"""
# TODO : desc
"""
function qqplot_std_data(fm::MaximumLikelihoodEVA{BlockMaxima{T}} where T<:Distribution)::DataFrame

    z = standardize(fm)

    y, p = ecdf(z)

    return DataFrame(Model = quantile.(Gumbel(), p), Empirical = y)

end

"""
# TODO : desc
"""
function qqplot_std_data(fm::MaximumLikelihoodEVA{ThresholdExceedance}, threshold::Real)::DataFrame

    z = standardize(fm, threshold)

    y, p = ecdf(z)

    return DataFrame(Model = quantile.(Exponential(), p), Empirical = y)

end

"""
# TODO : desc
"""
function qqplot_std(df::DataFrame)::Plot

    return plot(df, x=:Model, y=:Empirical, Geom.point, Geom.abline(color="red", style=:dash),
        Guide.xlabel("Model"), Guide.ylabel("Empirical"), Guide.title("Residual Quantile Plot"))

end

"""
# TODO : desc
"""
function qqplot_std(fm::MaximumLikelihoodEVA{BlockMaxima{T}} where T<:Distribution)::Plot

    df = qqplot_std_data(fm)

    return qqplot_std(df)

end

"""
# TODO : desc
"""
function qqplot_std(fm::MaximumLikelihoodEVA{ThresholdExceedance}, threshold::Real)::Plot

    df = qqplot_std_data(fm, threshold)

    return qqplot_std(df)

end

"""
# TODO : Desc
"""
function diagnosticplots_std(fm::MaximumLikelihoodEVA{BlockMaxima{T}} where T<:Distribution)::Gadfly.Compose.Context
    probplot = probplot_std(fm)
    qqplot = qqplot_std(fm)

    return hstack(probplot, qqplot)

end

"""
# TODO : Desc
"""
function diagnosticplots_std(fm::MaximumLikelihoodEVA{ThresholdExceedance}, threshold::Real)::Gadfly.Compose.Context
    probplot = probplot_std(fm, threshold)
    qqplot = qqplot_std(fm, threshold)

    return hstack(probplot, qqplot)

end
