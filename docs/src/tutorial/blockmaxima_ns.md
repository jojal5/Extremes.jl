
# Non-Stationary Block Maxima Model

In the non-stationary block maxima model, the GEV parameters are allowed to be a function of covariates:
- ``μ = X₁ × β₁``
- ``ϕ = X₂ × β₂``
- ``ξ = X₃ × β₃``
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
data = load("fremantle")
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

As show by Coles (2001), the best model is the one where the location parameter varies as a linear function of the year and the SOI, the `fm₂` in the present section. The approximate covariance matrix of the parameter estimates for this model can be obtained with the [`parametervar`](@ref) function:
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

Since the model parameters vary in time, the quantiles also vary in time. Therefore, a *T*-year return level can be estimated for each year. This set of return levels are referred to as *effective return levels* as proposed by Katz *et al.* (2002)[^1].

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
set_default_plot_size(12cm, 8cm)
plot(df, x=:Year, y=:r, ymin=:rmin, ymax=rmax, Geom.line, Geom.ribbon,
    Coord.cartesian(xmin=1895, xmax=1990), Guide.xticks(ticks=1895:10:1990),
    Guide.ylabel("100-year Effective Return Level"))

```

## Bayesian Inference

Most functions described in the previous sections also work in the Bayesian context.

### GEV parameter estimation

The Bayesian GEV parameter estimation is performed with the [`gevfitbayes`](@ref) function:

```@repl portpirie
fm = gevfitbayes(data, :SeaLevel)
```

!!! note "Prior"

    Currently, only the improper uniform prior is implemented, *i.e.*
    \\[ f_{(μ,ϕ,ξ)}(μ,ϕ,ξ) ∝ 1. \\]
    It yields to a proper posterior as long as the sample size is larger than 3 ([Northrop and Attalides, 2016](https://www.jstor.org/stable/24721296?seq=1)).

!!! note "Sampling scheme"

    Currently, the No-U-Turn Sampler extension ([Hoffman and Gelman, 2014](http://jmlr.org/papers/v15/hoffman14a.html)) to Hamiltonian Monte Carlo ([Neel, 2011, Chapter 5](https://www.mcmchandbook.net/)) is implemented for simulating an autocorrelated sample from the posterior distribution.

The generated sample from the posterior distribution is contained in the field `sim` of the fitted structure. It is an object of type *Chains* from the [*Mamba.jl*](https://mambajl.readthedocs.io/en/latest/index.html) package.

Credible intervals for the parameters are obtained with the [`cint`](@ref) function:
```@repl portpirie
cint(fm)
```


## Inference based on the probability weighted moments

Most functions described in the previous sections also work for the model fitted with the probability weighted moments.

### GEV parameter estimation

The parameter estimation based on the probability weighted moments is performed with the [`gevfitpwm`](@ref) function:

```@repl portpirie
fm = gevfitpwm(data, :SeaLevel)
```

The approximate covariance matrix of the parameter estimates using a bootstrap procedure can be obtained with the [`parametervar`](@ref) function:
```@repl portpirie
parametervar(fm)
```

Confidence intervals on the parameter estimates using a bootstrap procedure can be obtained with the [`cint`](@ref) function:
```@repl portpirie
cint(fm)
```


[^1]: Katz, R. W., M. B. Parlange, and P. Naveau (2002), Statistics of extremes in hydrology, Adv. Water Resour., 25, 1287–1304.
