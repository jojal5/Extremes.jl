
function standardize(y::Real, μ::Real, σ::Real, ξ::Real)::Real

    if ξ ≈ 0
        z = (y-μ)/σ
    else
        z = 1 / ξ * log( 1 + ξ/σ * ( y - μ ) )
    end

    return z

end


function standardize(fm::MaximumLikelihoodEVA)::Vector{<:Real}

    y = fm.model.data.value
    d = getdistribution(fm)

    return standardize.(y, location.(d), scale.(d), shape.(d))

end


function standardize(fm::BayesianEVA)::Vector{<:Real}

    θ̂ = findposteriormode(fm)
    dist = getdistribution(fm.model, θ̂)

    μ = location.(dist)
    σ = scale.(dist)
    ξ = shape.(dist)

    y = fm.model.data.value

    z = standardize.(y, μ, σ, ξ)

    return z
end



function standarddist(::BlockMaxima{T})::Distribution where T
    return Gumbel()
end


function standarddist(::ThresholdExceedance)::Distribution
    return Exponential()
end


function probplot_std_data(fm::fittedEVA)::DataFrame

    z = standardize(fm)

    y, p̂ = ecdf(z)

    return DataFrame(Model = cdf.(standarddist(fm.model), y), Empirical = p̂)

end


function qqplot_std_data(fm::fittedEVA)::DataFrame

    z = standardize(fm)

    y, p = ecdf(z)

    return DataFrame(Model = quantile.(standarddist(fm.model), p), Empirical = y)

end



function histplot_std_data(fm::fittedEVA)::Dict

    dist = standarddist(fm.model)

    z = standardize(fm)
    n = length(z)
    nbin = Int64(ceil(sqrt(n)))

    zmin = quantile(dist, 1/1000)
    zmax = quantile(dist, 1 - 1/1000)
    zp = range(zmin, zmax, length=1000)

    return Dict(:h => DataFrame(Data = z), :d => DataFrame(DataRange = zp, Density = pdf.(dist, zp)),
        :nbin => nbin, :xmin => zmin, :xmax => zmax)

end
