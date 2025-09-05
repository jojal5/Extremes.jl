"""
    probplot(fm::AbstractFittedExtremeValueModel)

Probability plot
"""
function probplot(fm::AbstractFittedExtremeValueModel; major_label_font_size=20pt, minor_label_font_size=16pt)

    if Extremes.getcovariatenumber(fm.model) > 0
        df = probplot_std_data(fm)
        plotTitle = "Residual Probability Plot"
    else
        df = probplot_data(fm)
        plotTitle = "Probability Plot"
    end

    return Gadfly.plot(df, x=:Model, y=:Empirical, Geom.point, Geom.abline(color="red", style=:dash),
        Guide.xlabel("Model"), Guide.ylabel("Empirical"), Guide.title(plotTitle),
        Gadfly.Theme(discrete_highlight_color=c->nothing, major_label_font_size=major_label_font_size, minor_label_font_size=minor_label_font_size))

end


"""
    qqplot(fm::AbstractFittedExtremeValueModel)

Quantile-Quantile plot

See also [`qqplotci`](@ref).   
"""
function qqplot(fm::AbstractFittedExtremeValueModel; major_label_font_size=20pt, minor_label_font_size=16pt)

    if Extremes.getcovariatenumber(fm.model) > 0
        df = qqplot_std_data(fm)
        plotTitle = "Residual Quantile Plot"
    else
        df = qqplot_data(fm)
        plotTitle = "Quantile Plot"
    end

    return Gadfly.plot(df, x=:Model, y=:Empirical, Geom.point, Geom.abline(color="red", style=:dash),
        Guide.xlabel("Model"), Guide.ylabel("Empirical"), Guide.title(plotTitle),
        Gadfly.Theme(discrete_highlight_color=c->nothing, major_label_font_size=major_label_font_size, minor_label_font_size=minor_label_font_size))

end


"""
    qqplotci(fm::AbstractFittedExtremeValueModel, α::Real=.05)

Quantile-Quantile plot along with the confidence/credible interval of level `1-α`.

## Note
This function is currently only available for stationary models.

See also [`returnlevelplotci`](@ref) and [`qqplot`](@ref).   

## Example

```@example
using Distributions, Extremes

pd = GeneralizedExtremeValue(0,1,0)
y = rand(pd, 300)
fm = gevfit(y)

qqplotci(fm)
```
 
"""
function qqplotci(fm::AbstractFittedExtremeValueModel, α::Real=.05; major_label_font_size=20pt, minor_label_font_size=16pt)
    @assert 0 < α < 1 "the level should be in (0,1)." 
    @assert Extremes.getcovariatenumber(fm.model) == 0 "adding confidence intervals is currently only available for stationary models."

    df = qqplot_data(fm)

    q, p = Extremes.ecdf(fm.model.data.value)

    q_inf = Float64[]
    q_sup = Float64[]

    for pᵢ in p
        c = cint(returnlevel(fm, 1 / (1 - pᵢ)), 1-α)[]
        push!(q_inf, c[1])
        push!(q_sup, c[2])
    end

    df[:,:Inf] = q_inf
    df[:,:Sup] = q_sup

    l1 = layer(df, x=:Model, y=:Empirical, Geom.point, Geom.abline(color="black", style=:dash), 
        Gadfly.Theme(default_color="black", discrete_highlight_color=c->nothing, major_label_font_size=major_label_font_size, minor_label_font_size=minor_label_font_size))
    l2 = layer(df, x=:Model, ymin=:Inf, ymax=:Sup, Geom.ribbon, Gadfly.Theme(lowlight_color=c->"lightgray", major_label_font_size=major_label_font_size, minor_label_font_size=minor_label_font_size))

    return Gadfly.plot(l1,l2, Guide.xlabel("Model"), Guide.ylabel("Empirical"))
end



"""
    returnlevelplot(fm::AbstractFittedExtremeValueModel)

Return level plot
"""
function returnlevelplot(fm::AbstractFittedExtremeValueModel; major_label_font_size=20pt, minor_label_font_size=16pt)

    if Extremes.getcovariatenumber(fm.model) > 0
        @warn "this graphic is not optimized for non-stationary model; plot ignored."
        return plot()
    else
        df = returnlevelplot_data(fm)
        l1 = layer(df, x=:Period, y=:Level,Geom.line, Gadfly.Theme(default_color="red", line_style=[:dash], major_label_font_size=major_label_font_size, minor_label_font_size=minor_label_font_size))
        l2 = layer(df, x=:Period, y=:Data, Geom.point)
        return Gadfly.plot(l1,l2, Scale.x_log10, Guide.xlabel("Return Period"), Guide.ylabel("Return Level"),
            Guide.title("Return Level Plot"), Gadfly.Theme(discrete_highlight_color=c->nothing, major_label_font_size=major_label_font_size, minor_label_font_size=minor_label_font_size))
    end
end


"""
    returnlevelplotci(fm::AbstractFittedExtremeValueModel, α::Real=.05)

Return level plot along with the confidence/credible interval of level `1-α`.

## Note
This function is currently only available for stationary models.

See also [`returnlevelplotci`](@ref) and [`qqplot`](@ref).   

## Example

```@example
using Distributions, Extremes

pd = GeneralizedExtremeValue(0,1,0)
y = rand(pd, 300)
fm = gevfit(y)

returnlevelplotci(fm)
```
"""
function returnlevelplotci(fm::AbstractFittedExtremeValueModel, α::Real=.05; major_label_font_size=20pt, minor_label_font_size=16pt)
    @assert 0 < α < 1 "the level should be in (0,1)." 
    @assert Extremes.getcovariatenumber(fm.model) == 0 "adding confidence intervals is currently only available for stationary models."

    df = returnlevelplot_data(fm)

    q, p = Extremes.ecdf(fm.model.data.value)

    q_inf = Float64[]
    q_sup = Float64[]
    
    for pᵢ in p
        c = cint(returnlevel(fm, 1 / (1 - pᵢ)), 1-α)[]
        push!(q_inf, c[1])
        push!(q_sup, c[2])
    end
    
    df[:,:Inf] = q_inf
    df[:,:Sup] = q_sup

    l1 = layer(df, x=:Period, y=:Level, Geom.line, Gadfly.Theme(default_color="black", line_style=[:dash], major_label_font_size=major_label_font_size, minor_label_font_size=minor_label_font_size))
    l2 = layer(df, x=:Period, y=:Data, Geom.point, Gadfly.Theme(default_color="black", discrete_highlight_color=c->nothing, major_label_font_size=major_label_font_size, minor_label_font_size=minor_label_font_size))
    l3 = layer(df, x=:Period, ymin=:Inf, ymax=:Sup, Geom.ribbon, Gadfly.Theme(lowlight_color=c->"lightgray", major_label_font_size=major_label_font_size, minor_label_font_size=minor_label_font_size))

    return Gadfly.plot(l1,l2,l3, Scale.x_log10, Guide.xlabel("Return Period"), Guide.ylabel("Return Level"),
        Guide.title("Return Level Plot"))

end



"""
    histplot(fm::AbstractFittedExtremeValueModel)

Histogram plot
"""
function histplot(fm::AbstractFittedExtremeValueModel; major_label_font_size=20pt, minor_label_font_size=16pt)

    if Extremes.getcovariatenumber(fm.model) > 0
        df = histplot_std_data(fm)
        plotTitle = "Residual Density Plot"
    else
        df = histplot_data(fm)
        plotTitle = "Density Plot"
    end

    h = layer(df[:h], x = :Data, Geom.histogram(bincount=df[:nbin], density=true), Gadfly.Theme(major_label_font_size=major_label_font_size, minor_label_font_size=minor_label_font_size))
    d = layer(df[:d], x = :DataRange, y = :Density, Geom.line, Gadfly.Theme(default_color="red", major_label_font_size=major_label_font_size, minor_label_font_size=minor_label_font_size))

    return Gadfly.plot(d,h, Gadfly.Coord.cartesian(xmin = df[:xmin], xmax = df[:xmax]), Guide.xlabel("Data"), Guide.ylabel("Density"), Guide.title(plotTitle), Gadfly.Theme(major_label_font_size=major_label_font_size, minor_label_font_size=minor_label_font_size))

end


"""
    diagnosticplots(fm::AbstractFittedExtremeValueModel)

Diagnostic plots
"""
function diagnosticplots(fm::AbstractFittedExtremeValueModel; title="", major_label_font_size=20pt, minor_label_font_size=16pt)::Gadfly.Compose.Context

    f1 = probplot(fm)
    f2 = qqplot(fm)
    f3 = histplot(fm)

    if Extremes.getcovariatenumber(fm.model) > 0
        f4 = Gadfly.plot()
    else
        f4 = returnlevelplot(fm)
    end

    return Gadfly.title(gridstack([f1 f2; f3 f4]), title, Gadfly.Compose.fontsize(major_label_font_size), fill(colorant"lightgray"))

end


"""
    mrlplot(y::Vector{<:Real}, steps::Int = 100)

Mean residual plot

See also [`mrlplot_data`](@ref).
"""
function mrlplot(y::Vector{<:Real}, steps::Int=100; major_label_font_size=20pt, minor_label_font_size=16pt)::Plot

    df = mrlplot_data(y, steps)

    p = Gadfly.plot(df, x = :Threshold, y = :mrl, ymin = :lbound, ymax = :ubound,
        Geom.line, Geom.ribbon, Guide.ylabel("Mean Residual Life"), Gadfly.Theme(major_label_font_size=major_label_font_size, minor_label_font_size=minor_label_font_size))

    return p
end
