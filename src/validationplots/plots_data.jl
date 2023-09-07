"""
    probplot_data(fm::fittedModel)

Return the probability plot data in a DataFrame.
"""
function probplot_data end

"""
    qqplot_data(fm::fittedModel)

Return the quantile-quantile plot data in a DataFrame.
"""
function qqplot_data end

"""
    returnlevelplot_data(fm::fittedModel)

Return the return level plot data in a DataFrame.
"""
function returnlevelplot_data end

"""
    histplot_data(fm::fittedModel)

Return the histogram plot data in a Dictionary.
"""
function hisplot_data end


function probplot_data(fm::BayesianAbstractExtremeValueModel)::DataFrame

    checkstationarity(fm.model)

    y, p̂ = ecdf(fm.model.data.value)

    dist = getdistribution(fm)

    p_sim = Array{Float64}(undef, length(dist), length(y))

    i = 1

    for d in eachslice(dist, dims=1)
        p_sim[i,:] = cdf.(d, y)
        i +=1
    end

    p = vec(mean(p_sim, dims=1))

    return DataFrame(Model = p, Empirical = p̂)

end


function probplot_data(fm::MaximumLikelihoodAbstractExtremeValueModel)::DataFrame

    checkstationarity(fm.model)

    y, p̂ = ecdf(fm.model.data.value)

    dist = getdistribution(fm.model, fm.θ̂)[]

    p = cdf.(dist, y)

    return DataFrame(Model = p, Empirical = p̂)

end


function probplot_data(fm::pwmAbstractExtremeValueModel)::DataFrame

    checkstationarity(fm.model)

    y, p̂ = ecdf(fm.model.data.value)

    dist = getdistribution(fm.model, fm.θ̂)[]

    p = cdf.(dist, y)

    return DataFrame(Model = p, Empirical = p̂)

end



function qqplot_data(fm::BayesianAbstractExtremeValueModel)::DataFrame

    checkstationarity(fm.model)

    y, p = ecdf(fm.model.data.value)

    dist = getdistribution(fm)

    q_sim = Array{Float64}(undef, length(dist), length(y))

    i = 1

    for d in eachslice(dist, dims=1)
        q_sim[i,:] = quantile.(d, p)
        i +=1
    end

    q = vec(mean(q_sim, dims=1))

    return DataFrame(Model = q, Empirical = y)

end


function qqplot_data(fm::MaximumLikelihoodAbstractExtremeValueModel)::DataFrame

    checkstationarity(fm.model)

    y, p = ecdf(fm.model.data.value)

    n = length(y)

    q = Vector{Float64}(undef, n)

    for i in eachindex(p)
       q[i] = quantile(fm, p[i])[1]
    end

    return DataFrame(Model = q, Empirical = y)

end


function qqplot_data(fm::pwmAbstractExtremeValueModel)::DataFrame

    checkstationarity(fm.model)

    y, p = ecdf(fm.model.data.value)

    n = length(y)

    q = Vector{Float64}(undef, n)

    for i in eachindex(p)
       q[i] = quantile(fm, p[i])[1]
    end

    return DataFrame(Model = q, Empirical = y)

end


function returnlevelplot_data(fm::BayesianAbstractExtremeValueModel)::DataFrame

    checkstationarity(fm.model)

    y, p = ecdf(fm.model.data.value)

    T = 1 ./ (1 .- p)

    dist = getdistribution(fm)

    q_sim = Array{Float64}(undef, length(dist), length(y))

    i = 1
    for d in eachslice(dist, dims=1)
        q_sim[i,:] = quantile.(d, p)
        i +=1
    end

    q = vec(mean(q_sim, dims=1))

    return DataFrame(Data = y, Period = T, Level = q)

end



function returnlevelplot_data(fm::MaximumLikelihoodAbstractExtremeValueModel)::DataFrame

    checkstationarity(fm.model)

    df = qqplot_data(fm)

    y, p = ecdf(fm.model.data.value)

    T = 1 ./ (1 .- p)

    n = length(y)

    q = Vector{Float64}(undef, n)

    for i in eachindex(p)
       q[i] = quantile(fm, p[i])[1]
    end

    return DataFrame(Data = y, Period = T, Level = q)

end



function returnlevelplot_data(fm::pwmAbstractExtremeValueModel)::DataFrame

    checkstationarity(fm.model)

    y, p = ecdf(fm.model.data.value)

    T = 1 ./ (1 .- p)

    n = length(y)

    q = Vector{Float64}(undef, n)

    for i in eachindex(p)
       q[i] = quantile(fm, p[i])[1]
    end

    return DataFrame(Data = y, Period = T, Level = q)

end


function histplot_data(fm::BayesianAbstractExtremeValueModel)::Dict

    checkstationarity(fm.model)

    x = fm.model.data.value
    n = length(x)
    nbin = Int64(ceil(sqrt(n)))

    dist = getdistribution(fm)

    xmin = quantile(dist[1], 1/1000)
    xmax = quantile(dist[1], 1 - 1/1000)
    xp = range(xmin, xmax, length=1000)

    h_sim = Array{Float64}(undef, length(dist), length(xp))

    i = 1
    for d in eachslice(dist, dims=1)
        h_sim[i,:] = pdf.(d, xp)
        i +=1
    end

    h = vec(mean(h_sim, dims=1))

    return Dict(:h => DataFrame(Data = x), :d => DataFrame(DataRange = xp, Density = h),
        :nbin => nbin, :xmin => xmin, :xmax => xmax)

end


function histplot_data(fm::MaximumLikelihoodAbstractExtremeValueModel)::Dict

    checkstationarity(fm.model)

    dist = getdistribution(fm)[1]

    x = fm.model.data.value
    n = length(x)
    nbin = Int64(ceil(sqrt(n)))

    xmin = quantile(dist, 1/1000)
    xmax = quantile(dist, 1 - 1/1000)
    xp = range(xmin, xmax, length=1000)

    return Dict(:h => DataFrame(Data = x), :d => DataFrame(DataRange = xp, Density = pdf.(dist, xp)),
        :nbin => nbin, :xmin => xmin, :xmax => xmax)

end


function histplot_data(fm::pwmAbstractExtremeValueModel)::Dict

    checkstationarity(fm.model)

    dist = getdistribution(fm)[1]

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
    mrlplot_data(y::Vector{<:Real}, steps::Int = 100)::DataFrame

Compute the mean residual life from vector `y`.

The set of thresholds ranges from `minimum(y)` to the second-to-last larger
value in `steps` number of steps.
"""
function mrlplot_data(y::Vector{<:Real}, steps::Int = 100)::DataFrame

    @assert steps > 0 "the number of steps should be positive"

    umin = minimum(y)
    umax = sort(y)[end-2]

    threshold = range(umin, stop = umax, length = steps)

    df = DataFrame(
        Threshold = Float64[],
        mrl = Float64[],
        lbound = Float64[],
        ubound = Float64[],
    )

    for u in threshold
        z = y[y.>u] .- u

        n = length(z)
        m = mean(z)
        s = std(z)

        push!(df, [u; m; m - 1.96 * s / sqrt(n); m + 1.96 * s / sqrt(n)])
    end

    return df

end
