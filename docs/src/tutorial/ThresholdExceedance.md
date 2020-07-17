
# Threshold Exceedance Model

The stationary [`ThresholdExceedance`](@ref) model is illustrated using the daily rainfall accumulations at a location in south-west England from 1914 to 1962. This dataset was studied by Coles (2001) in Chapter 4. The daily rainfall are assume  **independent and identically distributed**.

```@setup rain
using Extremes, Dates, DataFrames, Distributions, Gadfly
```

The *Extremes.jl* package supports maximum likelihood inference, Bayesian inference and inference based on the probability weigthed moments. For the GEV parameter estimation, the following functions can be used:
- [`gpfitpwm`](@ref): estimation with the probability weighted moments;
- [`gpfit`](@ref): maximum likelihood estimation;
- [`gpfitbayes`](@ref): Bayesian estimation.

!!! note "Log-scale paremeter"

    These functions return the estimate of the log-scale parameter $\phi = \log \sigma$.

## Load the data

Loading the daily precipitations:
```@example rain
data = Extremes.dataset("rain")
first(data,5)
```

Plotting the data using the Gadfly package:
```@example rain
set_default_plot_size(14cm ,8cm) # hide
plot(data, x=:Date, y=:Rainfall, Geom.point, Theme(discrete_highlight_color=c->nothing))
```

## Threshold selection

A suitable threshold for the Peaks-Over-Threshold model can be chosen by examining the mean residual life plot. The mean residual life is expected to be a linear function of the threshold when the latter is high enough. The mean residual life plot can be plotted with the [`mrlplot`](@ref) function:
```@example rain
set_default_plot_size(14cm ,8cm) # hide
mrlplot(data[:,:Rainfall])
```

As concluded by Coles (2001, Chapter 4), a reasonable threshold is 30 *mm*.
```@example rain
threshold = 30.0
nothing # hide
```

## Exceedances extraction

Parameter estimation of the Generalized Pareto distribution in *Extremes.jl* is performed using the threshold exceedances previously extracted. The support of the exceedances given in the fit function is therefore ``(0,∞)``.

For the *Rainfall* example, let's extract the threshold exceedances.

Identify first the threshold exceedances:
```@example rain
threshold = 30.0
df = filter(row -> row.Rainfall > threshold, data)
first(df, 5)
```

Retrieve the exceedances above the threshold:
```@example rain
df[!,:Rainfall] =  df[!,:Rainfall] .- threshold
rename!(df, :Rainfall => :Exceedance)
first(df, 5)
```

## Maximum likelihood inference


### GP parameter estimation

The Generalized Pareto maximum likelihood parameter estimates are computed with the [`gpfit`](@ref) function:

```@repl rain
fm = gpfit(df, :Exceedance)
```

!!! note
    In this example, the [`gpfit`](@ref) function is called using the data *DataFrame* structure as the first argument. The function can also be called using the vector of maxima as the first argument, *e.g.* `gpfit(df[:,:Exceedance])`.

The vector of the parameter estimates ``\hat\mathbf{\theta} = (ϕ̂,\, ξ̂)^\top`` is contained in the field `θ̂` of the structure `fm:<fittedEVA`.

The approximate covariance matrix of the parameter estimates can be obtained with the  [`parametervar`](@ref) function:
```@repl rain
parametervar(fm)
```

Confidence intervals for the parameters are obtained with the [`cint`](@ref) function:
```@repl rain
cint(fm)
```

For instance, the shape parameter 95% confidence interval is as follows:
```@repl rain
cint(fm)[2]
```

### Diagnostic plots

Several diagnostic plots for assessing the accuracy of the fitted GP distribution to the rainfall data are can be shown with the [`diagnosticplots`](@ref) function:

```@example rain
set_default_plot_size(21cm ,16cm)
diagnosticplots(fm)
```

The diagnostic plots consist in the probability plot (upper left panel), the quantile plot (upper right panel), the density plot (lower left panel) and the return level plot (lower right panel). These plots can be displayed separately using respectively the [`probplot`](@ref), [`qqplot`](@ref), [`histplot`](@ref) and [`returnlevelplot`](@ref) functions.


### Return level estimation

*T*-year return level estimate can be obtained using the [`returnlevel`](@ref) function. Along with the fitted Generalized Pareto distribution for the threshold exceedances, the following informations:
- the threshold;
- the number of total observation;
- the number of observation per year.
are also needed for estimating the *T*-year return level using the POT model.

For example, the 100-year return level for the rainfall POT model is computed as follows:
```@repl rain
nobs = size(data,1)
nobsperblock = 365
r = returnlevel(fm, threshold, nobs, nobsperblock, 100)
```

The return level can be accessed as follows:
```@repl rain
r.value
```

The corresponding confidence interval can be computed with the [`cint`](@ref) function:
```@repl rain
c = cint(r)
```

!!! note "Type-stable function"

    In this example of a stationary model, the function returns a unit dimension vector for the return level and a vector containing only one vector for the confidence interval. The reason is that the function always returns the same type in the stationary and non-stationary case. The function is therefore [type-stable](https://docs.julialang.org/en/v1/manual/performance-tips/index.html#Write-%22type-stable%22-functions-1) allowing better performance of code execution.  

To get the scalar return level in the stationary case, the following command can be used:
```@repl rain
r.value[]
```

To get the scalar confidence interval in the stationary case, the following command can be used:
```@repl rain
c[]
```


## Bayesian Inference

Most functions described in the previous sections also work in the Bayesian context.

### GP parameter estimation

The Bayesian GEV parameter estimation is performed with the [`gpfitbayes`](@ref) function:

```@repl rain
fm = gevfitbayes(df, :Exceedance)
```

!!! note "Prior"

    Currently, only the improper uniform prior is implemented, *i.e.*
    \\[ f_{(ϕ,ξ)}(ϕ,ξ) ∝ 1. \\]
    It yields to a proper posterior as long as the sample size is larger than 2 ([Northrop and Attalides, 2016](https://www.jstor.org/stable/24721296?seq=1)).

!!! note "Sampling scheme"

    Currently, the No-U-Turn Sampler extension ([Hoffman and Gelman, 2014](http://jmlr.org/papers/v15/hoffman14a.html)) to Hamiltonian Monte Carlo ([Neel, 2011, Chapter 5](https://www.mcmchandbook.net/)) is implemented for simulating an autocorrelated sample from the posterior distribution.

The generated sample from the posterior distribution is contained in the field `sim` of the fitted structure. It is an object of type *Chains* from the [*Mamba.jl*](https://mambajl.readthedocs.io/en/latest/index.html) package.

Credible intervals for the parameters are obtained with the [`cint`](@ref) function:
```@repl rain
cint(fm)
```


## Inference based on the probability weighted moments

Most functions described in the previous sections also work for the model fitted with the probability weighted moments.

### GP parameter estimation

The parameter estimation based on the probability weighted moments is performed with the [`gpfitpwm`](@ref) function:

```@repl rain
fm = gevfitpwm(df, :Exceedance)
```

The approximate covariance matrix of the parameter estimates using a bootstrap procedure can be obtained with the [`parametervar`](@ref) function:
```@repl rain
parametervar(fm)
```

Confidence intervals on the parameter estimates using a bootstrap procedure can be obtained with the [`cint`](@ref) function:
```@repl rain
cint(fm)
```
