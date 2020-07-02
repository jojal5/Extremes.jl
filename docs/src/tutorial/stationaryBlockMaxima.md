
# Block Maxima Model

The stationary [`BlockMaxima`](@ref) model is illustrated using the annual maximum sea-levels recorded at Port Pirie in South Australia from 1923 to 1987, studied by Coles (2001) in Chapter 3.

```@setup portpirie
using Extremes, DataFrames, Distributions, Gadfly
```

The *Extremes.jl* package supports maximum likelihood inference, Bayesian inference and inference based on the probability weigthed moments. For the GEV parameter estimation, the following functions can be used:
- [`gevfitpwm`](@ref): estimation with probability weighted moments;
- [`gevfit`](@ref): estimation with maximum likelihood;
- [`gevfitbayes`](@ref): estimation with the Bayesian method.

These functions return a `fittedEVA` type that can be used by all the other functions presented in this tutorial. The parameters estimates are contained in the field `θ̂` of this structure.

!!! note "Log-scale paremeter"

    These functions return the estimate of the log-scale parameter $\phi = \log \sigma$.


In this example, the data are contained in a *DataFrame*. The fit functions can be called using the DataFrame as the first argument and the data column symbol as the second argument.

## Load the data

Loading the annual maximum sea-levels at Port Pirie:
```@example portpirie
data = load("portpirie")
first(data,5)
```

The loaded data are contained in a Dataframe. The annual maxima can be shown as a function of the year using the Gadfly package:
```@example portpirie
set_default_plot_size(12cm, 8cm)
plot(data, x=:Year, y=:SeaLevel, Geom.line)
```

## Maximum likelihood inference


### GEV parameters estimation

The GEV parameter estimation with maximum likelihood is performed with the [`gevfit`](@ref) function:

```@repl portpirie
fm = gevfit(data, :SeaLevel)
```

The vector of the parameter estimates ``\hat\mathbf{\theta} = (μ̂,\, ϕ̂,\, ξ̂)^\top`` is contained in the field `θ̂` of the structure `fm:<fittedEVA`.

The approximate covariance matrix of the parameter estimates can be obtained with the  [`parametervar`](@ref) function:
```@repl portpirie
parametervar(fm)
```

Confidence intervals on the parameter estimates can be obtained with the [`cint`](@ref) function:
```@repl portpirie
cint(fm)
```

### Diagnostic plots

Several diagnostic plots for assessing the accuracy of the GEV model fitted to the Port Pirie data are can be shown with the [`diagnosticplots`](@ref) function:

```@example portpirie
set_default_plot_size(21cm ,16cm)
diagnosticplots(fm)
```

The diagnostic plots consist in the probability plot (upper left panel), quantile plot (upper right panel), return level plot (lower left panel) and the density plot (lower right panel). These plots can be displayed separately using respectively the functions [`probplot`](@ref), [`qqplot`](@ref), [`returnlevelplot`](@ref) and [`histplot`](@ref).


### Return level estimation

*T*-year return level estimate can be obtained using the function [`returnlevel`](@ref) on a `fittedEVA` object. The first argument is the fitted model, the second is the return period in years and the last one is the confidence level for computing the confidence interval.

For example, the 100-year return level for the Port Pirie block maxima model and the corresponding 95% confidence interval can be estimated with this commands:

```@repl portpirie
r = returnlevel(fm, 100, .95)
```

where the return value can be accessed with
```@repl portpirie
r.value
```

and where the corresponding confidence interval can be accessed with
```@repl portpirie
r.cint
```

!!! note "Type-stable function"

    In this example of a stationary model, the function returns a unit dimension vector for the return level and a vector containing only one vector for the confidence interval. The reason is that the function always returns the same type in the stationary and non-stationary case. The function is therefore [type-stable](https://docs.julialang.org/en/v1/manual/performance-tips/index.html#Write-%22type-stable%22-functions-1) allowing better performance of code execution.  

To get the scalar return level in the stationary case, the following command can be used:
```@repl portpirie
r.value[]
```

To get the scalar confidence interval in the stationary case, the following command can be used:
```@repl portpirie
r.cint[]
```


## Bayesian Inference

### GEV parameters estimation

The GEV parameter estimation with the Bayesian method is performed with the [`gevfitbayes`](@ref) function:

```@repl portpirie
fm = gevfitbayes(data, :SeaLevel)
```

!!! note "Prior"

    Currently, only the improper uniform prior is implemented, *i.e.*
    \\[ f_{(μ,ϕ,ξ)}(μ,ϕ,ξ) ∝ 1. \\]
    It yields to a proper posterior as long as the sample size is larger than 3 ([Northrop and Attalides, 2016](https://www.jstor.org/stable/24721296?seq=1)).

!!! note "Sampling scheme"

    Currently, the No-U-Turn Sampler extension ([Hoffman and Gelman, 2014](http://jmlr.org/papers/v15/hoffman14a.html)) to Hamiltonian Monte Carlo ([Neel, 2011, Chapter 5](https://www.mcmchandbook.net/)) is implemented for simulating an autocorrelated sample from the posterior distribution.


The approximate covariance matrix of the parameter estimates can be obtained with the  [`parametervar`](@ref) function:
```@repl portpirie
parametervar(fm)
```

Confidence intervals on the parameter estimates can be obtained with the [`cint`](@ref) function:
```@repl portpirie
cint(fm)
```

### Diagnostic plots

Several diagnostic plots for assessing the accuracy of the GEV model fitted to the Port Pirie data are can be shown with the [`diagnosticplots`](@ref)function:

```@example portpirie
set_default_plot_size(21cm ,16cm)
diagnosticplots(fm)
```

The diagnostic plots consist in the probability plot (upper left panel), quantile plot (upper right panel), return level plot (lower left panel) and the density plot (lower right panel). These plots can be displayed separately using respectively the functions [`probplot`](@ref), [`qqplot`](@ref), [`returnlevelplot`](@ref) and [`histplot`](@ref).


### Return level estimation

*T*-year return level estimate can be obtained using the function [`returnlevel`](@ref) on a `fittedEVA` object. The first argument is the fitted model, the second is the return period in years and the last one is the confidence level for computing the confidence interval.

For example, the 100-year return level for the Port Pirie block maxima model and the corresponding 95% confidence interval can be estimated with this commands:

```@repl portpirie
r = returnlevel(fm, 100, .95)
```

where the return value can be accessed with
```@repl portpirie
r.value
```

and where the corresponding confidence interval can be accessed with
```@repl portpirie
r.cint
```

!!! note "Type-stable function"

    In this example of a stationary model, the function returns a unit dimension vector for the return level and a vector containing only one vector for the confidence interval. The reason is that the function always returns the same type in the stationary and non-stationary case. The function is therefore [type-stable](https://docs.julialang.org/en/v1/manual/performance-tips/index.html#Write-%22type-stable%22-functions-1) allowing better performance of code execution.  

To get the scalar return level in the stationary case, the following command can be used:
```@repl portpirie
r.value[]
```

To get the scalar confidence interval in the stationary case, the following command can be used:
```@repl portpirie
r.cint[]
```



## Inference based on the probability weighted moments

### GEV parameters estimation

The parameter estimation with the probability weighted moments method is performed with the [`gevfitpwm`](@ref) function:

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

### Diagnostic plots

Several diagnostic plots for assessing the accuracy of the GEV model fitted to the Port Pirie data are can be shown with the [`diagnosticplots`](@ref)function:

```@example portpirie
set_default_plot_size(21cm ,16cm)
diagnosticplots(fm)
```

The diagnostic plots consist in the probability plot (upper left panel), quantile plot (upper right panel), return level plot (lower left panel) and the density plot (lower right panel). These plots can be displayed separately using respectively the functions [`probplot`](@ref), [`qqplot`](@ref), [`returnlevelplot`](@ref) and [`histplot`](@ref).


### Return level estimation

*T*-year return level estimate can be obtained using the function [`returnlevel`](@ref) on a `fittedEVA` object. The first argument is the fitted model, the second is the return period in years and the last one is the confidence level for computing the confidence interval.

For example, the 100-year return level for the Port Pirie block maxima model and the corresponding 95% confidence interval can be estimated with this commands:

```@repl portpirie
r = returnlevel(fm, 100, .95)
```

where the return value can be accessed with
```@repl portpirie
r.value
```

and where the corresponding confidence interval can be accessed with
```@repl portpirie
r.cint
```

!!! note "Type-stable function"

    In this example of a stationary model, the function returns a unit dimension vector for the return level and a vector containing only one vector for the confidence interval. The reason is that the function always returns the same type in the stationary and non-stationary case. The function is therefore [type-stable](https://docs.julialang.org/en/v1/manual/performance-tips/index.html#Write-%22type-stable%22-functions-1) allowing better performance of code execution.  

To get the scalar return level in the stationary case, the following command can be used:
```@repl portpirie
r.value[]
```

To get the scalar confidence interval in the stationary case, the following command can be used:
```@repl portpirie
r.cint[]
```
