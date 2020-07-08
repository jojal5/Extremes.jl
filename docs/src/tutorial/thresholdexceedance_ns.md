
# Non-Stationary Threshold Exceedance Model

In the non-stationary threshold exceedance model, the GP parameters are allowed to be functions of multiple covariates:

\\[ ϕ = X₂ × β₂ \\\  ξ = X₃ × β₃ \\]

where ``(X₂,β₂)`` and ``(X₃,β₃)`` are respectively the design matrix and the corresponding coefficient parameter vector of ``ϕ`` and ``ξ``.

!!! note "Intercept"

    An intercept is included in all parameter functions by default.

The non-stationary [`ThresholdExceedance`](@ref) model is illustrated using the daily rainfall accumulations at a location in south-west England from 1914 to 1962, studied by Coles (2001) in Chapter 6.

```@setup rainfall
using Extremes, DataFrames, Dates, Distributions, Gadfly
```

### Load the data

Loading the daily rainfall accumulations:
```@example rainfall
data = load("rain")
first(data,5)
```

Extract the exceedances over the threshold of 30 *mm*:
```@example rainfall
threshold = 30.0
df = filter(row -> row.Rainfall > threshold, data)
df[!,:Exceedance] = df[:,:Rainfall] .- threshold
df[!,:Year] = year.(df[:,:Date])
set_default_plot_size(12cm, 8cm) # hide
plot(df, x=:Date, y=:Exceedance, Geom.point)
```
Non-stationary parameter estimation can be performed either by maximum likelihood or by the Bayesian approach. Probability weighted moment estimation cannot be used in the non-stationary case.

## Maximum likelihood inference

### GP parameter estimation

The GP parameter estimation with maximum likelihood is performed with the [`gpfit`](@ref) function. The parameter estimate vector ``\mathbf{θ̂} = (\mathbf{β̂₂},\, \mathbf{β̂₃})^\top`` is contained in the field `θ̂` of the returned structure.

Several non-stationary model can be fitted.

#### The stationary model
```@repl rainfall
fm₀ = gpfit(df, :Exceedance)
```
#### The logscale parameter varying as a linear function of the year
```@repl rainfall
fm₁ = gpfit(df, :Exceedance, logscalecovid = [:Year])
```

Confidence intervals for the parameters are obtained with the [`cint`](@ref) function:
```@repl rainfall
cint(fm₁)
```

In particular, the 95% confidence interval for the rise in the log-scale parameter per year is as follows:
```@repl rainfall
cint(fm₁)[2]
```

### Diagnostic plots

Several diagnostic plots for assessing the accuracy of the GP model fitted to the rainfall data can be shown with the [`diagnosticplots`](@ref) function:

```@example rainfall
set_default_plot_size(21cm ,16cm)
diagnosticplots(fm₁)
```

The diagnostic plots consist in the residual probability plot (upper left panel), the residual quantile plot (upper right panel) and the residual density plot (lower left panel) of the standardized data (see Chapter 6 of Coles, 2001). These plots can be displayed separately using respectively the [`probplot`](@ref), [`qqplot`](@ref), [`histplot`](@ref) and [`returnlevelplot`](@ref) functions.


### Return level estimation

Since the model parameters vary in time, the quantiles also vary in time. Therefore, a *T*-year return level can be estimated for each year. This set of return levels are referred to as *effective return levels* as proposed by Katz *et al.* (2002)[^1].

The 100-year effective return levels for the `fm₁` model can be computed using the [`returnlevel`](@ref) function:
```@repl rainfall
nobs = size(data,1)
nobsperblock = 365
r = returnlevel(fm₁, threshold, nobs, nobsperblock, 100)
```

The effective return levels can be accessed as follows:
```@repl rainfall
r.value
```

The corresponding confidence interval can be computed with the [`cint`](@ref) function:
```@repl rainfall
c = cint(r)
```

The effective return levels along with their confidence intervals can be plotted as follows:

```@example rainfall
rmin = [c[i][1] for i in eachindex(c)]
rmax = [c[i][2] for i in eachindex(c)]
df_plot = DataFrame(Year = data[:,:Year], r = r.value, rmin = rmin, rmax = rmax)
nothing # hide
```

```@example rainfall
set_default_plot_size(12cm, 8cm)
plot(df_plot, x=:Year, y=:r, ymin=:rmin, ymax=rmax, Geom.line, Geom.ribbon,
    Coord.cartesian(xmin=1895, xmax=1990), Guide.xticks(ticks=1895:10:1990),
    Guide.ylabel("100-year Effective Return Level"))

```

## Bayesian Inference

Most functions described in the previous sections also work in the Bayesian context.

### GP parameter estimation

The Bayesian GP parameter estimation is performed with the [`gpfitbayes`](@ref) function:

```@repl rainfall
fm = gpfitbayes(df, :Exceedance, logscalecovid = [:Year])
```

!!! note "Prior"

    Currently, only the improper uniform prior is implemented, *i.e.*
    \\[ f_{(β₂,β₃)}(β₂,β₃) ∝ 1. \\]


[^1]: Katz, R. W., M. B. Parlange, and P. Naveau (2002), Statistics of extremes in hydrology, Adv. Water Resour., 25, 1287–1304.
