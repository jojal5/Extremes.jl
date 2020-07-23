"""
    probplot(fm::fittedEVA)

Probability plot
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
        Guide.xlabel("Model"), Guide.ylabel("Empirical"), Guide.title(plotTitle),
        Theme(discrete_highlight_color=c->nothing))

end


"""
    qqplot(fm::fittedEVA)

Quantile-Quantile plot
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
        Guide.xlabel("Model"), Guide.ylabel("Empirical"), Guide.title(plotTitle),
        Theme(discrete_highlight_color=c->nothing))

end


"""
    returnlevelplot(fm::fittedEVA)

Return level plot
"""
function returnlevelplot(fm::fittedEVA)::Plot

    if getcovariatenumber(fm.model) > 0
        @warn "this graphic is not optimized for non-stationary model; plot ignored."
        return plot()
    else
        df = returnlevelplot_data(fm)
        l1 = layer(df, x=:Period, y=:Level,Geom.line, Theme(default_color="red", line_style=[:dash]))
        l2 = layer(df, x=:Period, y=:Data, Geom.point)
        return plot(l1,l2, Scale.x_log10, Guide.xlabel("Return Period"), Guide.ylabel("Return Level"),
            Guide.title("Return Level Plot"), Theme(discrete_highlight_color=c->nothing))
    end
end


"""
    histplot(fm::fittedEVA)

Histogram plot
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
    diagnosticplots(fm::fittedEVA)

Diagnostic plots
"""
function diagnosticplots(fm::fittedEVA)::Gadfly.Compose.Context

    f1 = probplot(fm)
    f2 = qqplot(fm)
    f3 = histplot(fm)

    if getcovariatenumber(fm.model) > 0
        f4 = plot()
    else
        f4 = returnlevelplot(fm)
    end

    return gridstack([f1 f2; f3 f4])
end


"""
    mrlplot(y::Vector{<:Real}, steps::Int = 100)

Mean residual plot

See also [`mrlplot_data`](@ref).
"""
function mrlplot(y::Vector{<:Real}, steps::Int=100)::Plot

    df = mrlplot_data(y, steps)

    p = plot(df, x = :Threshold, y = :mrl, ymin = :lbound, ymax = :ubound,
        Geom.line, Geom.ribbon, Guide.ylabel("Mean Residual Life"))

    return p
end
