"""
# TODO : desc
"""
function probplot(fm::fittedEVA)::Plot

    if getcovariatenumber(fm.model) > 0
        df = probplot_std_data(fm)
        plotTitle = "Residual Probability Plot"
    else
        df = probplot_data(fm)
        plotTitle = "Probability Plot"
    end

    return plot(df, x=:Model, y=:Empirical, Geom.point, Geom.abline(color="red", style=:dash),
        Guide.xlabel("Model"), Guide.ylabel("Empirical"), Guide.title(plotTitle))

end


"""
# TODO : desc
"""
function probplot_data(fm::MaximumLikelihoodEVA)::DataFrame

    checkstationarity(fm.model)

    y, p̂ = ecdf(fm.model.data.value)

    dist = getdistribution(fm.model, fm.θ̂)[]

    p = cdf.(dist, y)

    return DataFrame(Model = p, Empirical = p̂)

end

"""
# TODO : desc
"""
function probplot_data(fm::BayesianEVA)::DataFrame

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



"""
# TODO : desc
"""
function qqplot(fm::fittedEVA)::Plot

    if getcovariatenumber(fm.model) > 0
        df = qqplot_std_data(fm)
        plotTitle = "Residual Quantile Plot"
    else
        df = qqplot_data(fm)
        plotTitle = "Quantile Plot"
    end

    return plot(df, x=:Model, y=:Empirical, Geom.point, Geom.abline(color="red", style=:dash),
        Guide.xlabel("Model"), Guide.ylabel("Empirical"), Guide.title(plotTitle))

end


"""
# TODO : desc
"""
function qqplot_data(fm::pwmEVA)::DataFrame

    checkstationarity(fm.model)

    y, p = ecdf(fm.model.data.value)

    n = length(y)

    q = Vector{Float64}(undef, n)

    for i in eachindex(p)
       q[i] = quantile(fm, p[i])[1]
    end

    return DataFrame(Model = q, Empirical = y)

end

"""
# TODO : desc
"""
function qqplot_data(fm::MaximumLikelihoodEVA)::DataFrame

    checkstationarity(fm.model)

    y, p = ecdf(fm.model.data.value)

    n = length(y)

    q = Vector{Float64}(undef, n)

    for i in eachindex(p)
       q[i] = quantile(fm, p[i])[1]
    end

    return DataFrame(Model = q, Empirical = y)

end

"""
# TODO : desc
"""
function qqplot_data(fm::BayesianEVA)::DataFrame

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


"""
# TODO : desc
"""
function returnlevelplot(fm::fittedEVA)::Plot

    if getcovariatenumber(fm.model) > 0
        @warn "this graphic is not optimized for non-stationary model; plot ignored."
        return plot()
    else
        df = returnlevelplot_data(fm)
        l1 = layer(df, x=:Period, y=:Level,Geom.line, Theme(default_color="red", line_style=[:dash]))
        l2 = layer(df, x=:Period, y=:Data, Geom.point)
        return plot(l1,l2, Scale.x_log10, Guide.xlabel("Return Period"), Guide.ylabel("Return Level"), Guide.title("Return Level Plot"))
    end
end

"""
# TODO : desc
"""
function returnlevelplot_data(fm::pwmEVA)::DataFrame

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

"""
# TODO : desc
"""
function returnlevelplot_data(fm::MaximumLikelihoodEVA)::DataFrame

    df = qqplot_data(fm)

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

"""
# TODO : desc
"""
function returnlevelplot_data(fm::BayesianEVA)::DataFrame

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


"""
# TODO : desc
"""
function histplot(fm::fittedEVA)::Plot

    if getcovariatenumber(fm.model) > 0
        df = histplot_std_data(fm)
        plotTitle = "Residual Density Plot"
    else
        df = histplot_data(fm)
        plotTitle = "Density Plot"
    end

    h = layer(df[:h], x = :Data, Geom.histogram(bincount=df[:nbin], density=true))
    d = layer(df[:d], x = :DataRange, y = :Density, Geom.line, Theme(default_color="red") )

    return plot(d,h, Coord.cartesian(xmin = df[:xmin], xmax = df[:xmax]), Guide.xlabel("Data"), Guide.ylabel("Density"), Guide.title(plotTitle))

end

"""
# TODO : desc
"""
function histplot_data(fm::pwmEVA)::Dict

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
# TODO : desc
"""
function histplot_data(fm::MaximumLikelihoodEVA)::Dict

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
# TODO : desc
"""
function histplot_data(fm::BayesianEVA)::Dict

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


"""
# TODO : desc
"""
function diagnosticplots(fm::fittedEVA)::Gadfly.Compose.Context

    f1 = probplot(fm)
    f2 = qqplot(fm)
    f3 = histplot(fm)

    if getcovariatenumber(fm.model) > 0
        f4 = plot()
    else
        f4 = histplot(fm)
    end

    return gridstack([f1 f2; f3 f4])
end


"""
    mrlplot_data(y::Vector{<:Real}, steps::Int = 100)::DataFrame

Compute the mean residual life from vector `y` using the set of thresholds from the `minimum(y)` value to the second-to-last larger value in `steps` steps.
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

"""
    mrlplot(y::Vector{<:Real}, steps::Int = 100)::Plot

Show the mean residual life from vector `y` using the set of thresholds from the `minimum(y)` value to the second-to-last larger value in `steps` steps.
"""
function mrlplot(y::Vector{<:Real}, steps::Int=100)::Plot

    df = mrlplot_data(y, steps)

    p = plot(df, x = :Threshold, y = :mrl, ymin = :lbound, ymax = :ubound,
        Geom.line, Geom.ribbon, Guide.ylabel("Mean Residual Life"))

    return p
end
