
# Non-Stationary Block Maxima Model

In the non-stationary block maxima model, the GEV parameters are allowed to be functions of multiple covariates:

\\[ μ = X₁ × β₁ \\\ ϕ = X₂ × β₂ \\\  ξ = X₃ × β₃ \\]

where ``(X₁,β₁)``, ``(X₂,β₂)`` and ``(X₃,β₃)`` are respectively the design matrix and the corresponding coefficient parameter vector of ``μ``, ``ϕ`` and ``ξ``.

!!! note "Intercept"

    An intercept is included in all parameter functions by default.

The non-stationary [`BlockMaxima`](@ref) model is illustrated using the annual maximum sea-levels recorded at Fremantle in West Australia from 1897 to 1989, studied by Coles (2001) in Chapter 6.

```@setup fremantle
using Extremes, DataFrames, Distributions, Gadfly
```

### Load the data

Loading the annual maximum sea-levels at Fremantle:
```@example fremantle
data = Extremes.dataset("fremantle")
first(data,5)
```

The annual maxima can be plotted as function of the year:
```@example fremantle
set_default_plot_size(12cm, 8cm) # hide
plot(data, x=:Year, y=:SeaLevel, Geom.line,
    Coord.cartesian(xmin=1895, xmax=1990), Guide.xticks(ticks=1895:10:1990))
```
and as function of the Southern Oscillation Index:
```@example fremantle
set_default_plot_size(12cm, 8cm) # hide
plot(data, x=:SOI, y=:SeaLevel, Geom.point)
```

Both variables can be included in the block maxima model. Parameter estimation can be performed either by maximum likelihood or by the Bayesian approach. Probability weighted moment estimation cannot be used in the non-stationary case.

## Maximum likelihood inference

### GEV parameter estimation

The GEV parameter estimation with maximum likelihood is performed with the [`gevfit`](@ref) function. The parameter estimate vector ``\mathbf{θ̂} = (\mathbf{β̂₁},\, \mathbf{β̂₂},\, \mathbf{β̂₃})^\top`` is contained in the field `θ̂` of the returned structure.

Several non-stationary model can be fitted.

#### The stationary model
```@repl fremantle
fm₀ = gevfit(data, :SeaLevel)
```
#### The location parameter varying as a linear function of the year
```@repl fremantle
fm₁ = gevfit(data, :SeaLevel, locationcovid = [:Year])
```
#### The location parameter varying as a linear function of the year and the SOI
```@repl fremantle
fm₂ = gevfit(data, :SeaLevel, locationcovid = [:Year, :SOI])
```
#### Both the location and logscale parameters varying as a linear function of the year and the SOI
```@repl fremantle
fm₃ = gevfit(data, :SeaLevel, locationcovid = [:Year, :SOI], logscalecovid = [:Year, :SOI])
```

As show by Coles (2001), the best model is the one where the location parameter varies as a linear function of the year and the SOI, the `fm₂` in the present section. 

The vector of the parameter estimates (location scale and shape) can be extracted with the function [`params`](@ref):
```@repl fremantle
params(fm₂)
```

The location parameter with the function [`location`](@ref):
```@repl fremantle
location(fm₂)
```

The scale parameter with the function [`Extremes.scale`](@ref):
```@repl fremantle
scale(fm₂)
```

The shape parameter with the function [`shape`](@ref):
```@repl fremantle
shape(fm₂)
```

The approximate covariance matrix of the parameter estimates for this model can be obtained with the [`parametervar`](@ref) function:
```@repl fremantle
parametervar(fm₂)
```

Confidence intervals for the parameters are obtained with the [`cint`](@ref) function:
```@repl fremantle
cint(fm₂)
```

In particular, the 95% confidence interval for the rise in annual maximum sea-levels per year is as follows:
```@repl fremantle
cint(fm₂)[2]
```

### Diagnostic plots

Several diagnostic plots for assessing the accuracy of the GEV model fitted to the Fremantle data can be shown with the [`diagnosticplots`](@ref) function:

```@example fremantle
set_default_plot_size(21cm ,16cm)
diagnosticplots(fm₂)
```

The diagnostic plots consist in the residual probability plot (upper left panel), the residual quantile plot (upper right panel) and the residual density plot (lower left panel) of the standardized data (see Chapter 6 of Coles, 2001). These plots can be displayed separately using respectively the [`probplot`](@ref), [`qqplot`](@ref), [`histplot`](@ref) and [`returnlevelplot`](@ref) functions.


### Return level estimation

Since the model parameters vary in time, the quantiles also vary in time. Therefore, a *T*-year return level can be estimated for each year. This set of return levels are referred to as *effective return levels* as proposed by [Katz *et al.* (2002)](https://www.sciencedirect.com/science/article/pii/S0309170802000568?casa_token=VLKUdsDORdoAAAAA:EaD9J7vxHQeVD0KVZ5zfdCfOosWO8IlS0-CwsJQb7ihtEj3W1vbHryflMuwFIPJsrcz9B8uFjA).

The 100-year effective return levels for the `fm₂` model can be computed using the [`returnlevel`](@ref) function:
```@repl fremantle
r = returnlevel(fm₂, 100)
```

The effective return levels can be accessed as follows:
```@repl fremantle
r.value
```

The corresponding confidence interval can be computed with the [`cint`](@ref) function:
```@repl fremantle
c = cint(r)
```

The effective return levels along with their confidence intervals can be plotted as follows:

```@example fremantle
rmin = [c[i][1] for i in eachindex(c)]
rmax = [c[i][2] for i in eachindex(c)]
df = DataFrame(Year = data[:,:Year], r = r.value, rmin = rmin, rmax = rmax)
nothing # hide
```

```@example fremantle
set_default_plot_size(12cm, 9cm)
plot(df, x=:Year, y=:r, ymin=:rmin, ymax=rmax, Geom.line, Geom.ribbon,
    Coord.cartesian(xmin=1895, xmax=1990), Guide.xticks(ticks=1895:10:1990),
    Guide.ylabel("100-year Effective Return Level"))

```

## Bayesian Inference

Most functions described in the previous sections also work in the Bayesian context.

### GEV parameter estimation

The Bayesian GEV parameter estimation is performed with the [`gevfitbayes`](@ref) function:

```@repl fremantle
fm = gevfitbayes(data, :SeaLevel, locationcovid = [:Year, :SOI])
```

!!! note "Prior"

    Currently, only the improper uniform prior is implemented, *i.e.*
    \\[ f_{(β₁,β₂,β₃)}(β₁,β₂,β₃) ∝ 1. \\]

## Inference for the non-stationary Gumbel distribution

The package aslo provides functions for the inference of the non-stationary Gumbel model. Documentation on the Gumbel model can be found here in the Block Maxima section.

### Example on the annual maximum sea-levels recorded at Fremantle


The location parameter varying as a linear function of the year and the SOI
```@repl fremantle
fm₂ = gumbelfit(data, :SeaLevel, locationcovid = [:Year, :SOI])
```

Confidence intervals for the parameters are obtained with the [`cint`](@ref) function:
```@repl fremantle
cint(fm₂)
```

The diagnostic plots for assessing the accuracy of the Gumbel model fitted to the Fremantle data can be shown with the [`diagnosticplots`](@ref) function:

```@example fremantle
set_default_plot_size(21cm ,16cm)
diagnosticplots(fm₂)
```

The 100-year effective return levels for the `fm₂` model can be computed using the [`returnlevel`](@ref) function:
```@repl fremantle
r = returnlevel(fm₂, 100)
```

The effective return levels can be accessed as follows:
```@repl fremantle
r.value
```

The corresponding confidence interval can be computed with the [`cint`](@ref) function:
```@repl fremantle
c = cint(r)
```

The effective return levels along with their confidence intervals can be plotted as follows:

```@example fremantle
rmin = [c[i][1] for i in eachindex(c)]
rmax = [c[i][2] for i in eachindex(c)]
df = DataFrame(Year = data[:,:Year], r = r.value, rmin = rmin, rmax = rmax)
nothing # hide
```

```@example fremantle
set_default_plot_size(12cm, 9cm)
plot(df, x=:Year, y=:r, ymin=:rmin, ymax=rmax, Geom.line, Geom.ribbon,
    Coord.cartesian(xmin=1895, xmax=1990), Guide.xticks(ticks=1895:10:1990),
    Guide.ylabel("100-year Effective Return Level"))

```
