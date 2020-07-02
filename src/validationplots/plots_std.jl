"""
# TODO : desc
"""
function standardize(y::Real, μ::Real, σ::Real, ξ::Real)::Real
    return 1 / ξ * log( 1 + ξ/σ * ( y - μ ) )

end

"""
# TODO : desc
"""
function standardize(fm::MaximumLikelihoodEVA)::Vector{<:Real}

    y = fm.model.data.value
    d = getdistribution(fm)

    return standardize.(y, location.(d), scale.(d), shape.(d))

end

"""
# TODO : desc
"""
function standardize(fm::BayesianEVA)::Vector{<:Real}

    θ̂ = Extremes.findposteriormode(fm)
    dist = Extremes.getdistribution(fm.model, θ̂)

    μ = location.(dist)
    σ = scale.(dist)
    ξ = shape.(dist)

    y = fm.model.data.value

    z = Extremes.standardize.(y, μ, σ, ξ)

    return z
end


"""
# TODO : desc
"""
function standarddist(::BlockMaxima)::Distribution
    return Gumbel()
end

"""
# TODO : desc
"""
function standarddist(::ThresholdExceedance)::Distribution
    return Exponential()
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
