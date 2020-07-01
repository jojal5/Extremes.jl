# Tutorial

This tutorial shows the functionalities of *Extremes.jl*. They are illustrated by reproducing some of the results shown by Coles (2001) in [An Introduction to Statistical Modeling of Extreme
 Values](http://www.springer.com/us/book/9781852334598).

Before executing this tutorial, make sure to have installed the following packages:
- *Extremes.jl* (of course)
- *DataFrames.jl* (for using the DataFrame type)
- *Distributions.jl* (for using probability distribution objects)
- *Gadfly.jl* (for plotting)

and import them using the following command:
 ```@repl
using Extremes, Dates, DataFrames, Distributions, Gadfly
```


## Model for stationary block maxima

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

### Load the data

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

### Maximum likelihood inference


#### GEV parameters estimation

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

#### Diagnostic plots

Several diagnostic plots for assessing the accuracy of the GEV model fitted to the Port Pirie data are can be shown with the [`diagnosticplots`](@ref) function:

```@example portpirie
set_default_plot_size(21cm ,16cm)
diagnosticplots(fm)
```

The diagnostic plots consist in the probability plot (upper left panel), quantile plot (upper right panel), return level plot (lower left panel) and the density plot (lower right panel). These plots can be displayed separately using respectively the functions [`probplot`](@ref), [`qqplot`](@ref), [`returnlevelplot`](@ref) and [`histplot`](@ref).


#### Return level estimation

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


### Bayesian Inference

#### GEV parameters estimation

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

#### Diagnostic plots

Several diagnostic plots for assessing the accuracy of the GEV model fitted to the Port Pirie data are can be shown with the [`diagnosticplots`](@ref)function:

```@example portpirie
set_default_plot_size(21cm ,16cm)
diagnosticplots(fm)
```

The diagnostic plots consist in the probability plot (upper left panel), quantile plot (upper right panel), return level plot (lower left panel) and the density plot (lower right panel). These plots can be displayed separately using respectively the functions [`probplot`](@ref), [`qqplot`](@ref), [`returnlevelplot`](@ref) and [`histplot`](@ref).


#### Return level estimation

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



### Inference based on the probability weighted moments

#### GEV parameters estimation

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

#### Diagnostic plots

Several diagnostic plots for assessing the accuracy of the GEV model fitted to the Port Pirie data are can be shown with the [`diagnosticplots`](@ref)function:

```@example portpirie
set_default_plot_size(21cm ,16cm)
diagnosticplots(fm)
```

The diagnostic plots consist in the probability plot (upper left panel), quantile plot (upper right panel), return level plot (lower left panel) and the density plot (lower right panel). These plots can be displayed separately using respectively the functions [`probplot`](@ref), [`qqplot`](@ref), [`returnlevelplot`](@ref) and [`histplot`](@ref).


#### Return level estimation

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






## Model for stationary threshold exceedances

The stationary [`ThresholdExceedance`](@ref) model is illustrated using the daily rainfall accumulations at a location in south-west England from 1914 to 1962. This dataset was studied by Coles (2001) in Chapter 4.

```@setup rain
using Extremes, Dates, DataFrames, Distributions, Gadfly
```

### Load the data

Loading the daily rainfall at a location in South-England:

```@example rain
data = load("rain")
first(data,5)
```

Plotting the data using the Gadfly package:
```@example rain
set_default_plot_size(14cm ,8cm)
plot(data, x=:Date, y=:Rainfall, Geom.point, Theme(discrete_highlight_color=c->nothing))
```

### Threshold selection

TODO

### GPD parameters estimation

Let's first identify the threshold exceedances:
```@example rain
threshold = 30.0
df = filter(row -> row.Rainfall > threshold, data)
first(df, 5)
```

Get the exceedances above the threshold:
```@example rain
df[!,:Rainfall] =  df[!,:Rainfall] .- threshold
rename!(df, :Rainfall => :Exceedance)
first(df, 5)
```

#### GP parameters estimation with probability weighted moments

The GP parameter estimation with probability weighted moments is performed as follows:

```@repl rain
fm = gpfitpwm(df, :Exceedance)
```

The approximate covariance matrix of the parameter estimates can be obtained with the function [`parametervar`](@ref):
```@repl rain
parametervar(fm)
```


#### GP parameters estimation with maximum likelihood

The GP parameter estimation with maximum likelihood is performed as follows:

```@repl rain
fm = gpfit(df, :Exceedance)
```

The approximate covariance matrix of the parameter estimates can be obtained with the function [`parametervar`](@ref):
```@repl rain
parametervar(fm)
```

#### GP parameters estimation with the Bayesian method

The GP parameter estimation with the Bayesian method is performed as follows:

```@repl rain
fm = gpfitbayes(df, :Exceedance)
```

!!! note "Prior"

    Currently, only the improper uniform prior is implemented, *i.e.*
    \\[ f_{(ϕ,ξ)}(ϕ,ξ) ∝ 1. \\]
    It yields to a proper posterior as long as the sample size is larger than 2 ([Northrop and Attalides, 2016](https://www.jstor.org/stable/24721296?seq=1)).

!!! note "Sampling scheme"

    Currently, the No-U-Turn Sampler extension ([Hoffman and Gelman, 2014](http://jmlr.org/papers/v15/hoffman14a.html)) to Hamiltonian Monte Carlo ([Neel, 2011, Chapter 5](https://www.mcmchandbook.net/)) is implemented for simulating an autocorrelated sample from the posterior distribution.


The approximate covariance matrix of the parameter estimates can be obtained with the function [`parametervar`](@ref):
```@repl rain
parametervar(fm)
```


### Return level estimation

With the [`ThresholdExceedance`](@ref) structure, the [`returnlevel`](@ref) function requires several arguments to calculate the *T*-year return level:
- the threshold value;
- the number of total observation (below and above the threshold);
- the number of observations per year;
- the return period *T*;
- the confidence level for computing the confidence interval.
The function uses the Peaks-Over-Threshold model definition (Coles, 2001, Chapter 4) for computing the *T*-year return level.

For the rainfall example, the 100-year return level can be estimated as follows:

```@example rain
fm = gpfit(df, :Exceedance)
r = returnlevel(fm, threshold, size(data,1), 365, 100, .95)
```

where the value can be accessed with
```@repl rain
r.value
```

and where the corresponding confidence interval can be accessed with
```@repl rain
r.cint
```

!!! note

    In this example of a stationary model, the function returns a unit dimension vector for the return level and a vector containing only one vector for the confidence interval. The reason is that the function always returns the same type in the stationary and non-stationary case. The function is therefore [type-stable](https://docs.julialang.org/en/v1/manual/performance-tips/index.html#Write-%22type-stable%22-functions-1) allowing better performance of code execution.  

To get the scalar return level in the stationary case, the following command can be used:
```@repl rain
r.value[]
```

To get the scalar confidence interval in the stationary case, the following command can be used:
```@repl rain
r.cint[]
```

### Probability weighted moments estimation  

Probability weighted moments estimation of the GEV parameters can also be performed by using the [`gevfitpwm`](@ref) function. All the methods also apply to the `pwmEVA` object.

```@repl rain
fm = gpfitpwm(df, :Exceedance)
```

### Bayesian estimation

Bayesian estimation of the GEV parameters can also be performed by using the [`gevfitbayes`](@ref) function. All the methods also apply to the `BayesianEVA object.

```@repl rain
fm = gpfitbayes(df, :Exceedance)
```



## Model for dependent data


The stationary [`ThresholdExceedance`](@ref) model is illustrated using the daily rainfall accumulations at a location in south-west England from 1914 to 1962. This dataset was studied by Coles (2001) in Chapter 4.

```@setup wooster
using Extremes, Dates, DataFrames, Distributions, Gadfly
```

### Load the data

Loading the daily rainfall at a location in South-England:

```@example wooster
data = load("wooster")
first(data,5)
```

Plotting the data using the Gadfly package:
```@example wooster
plot(data, x=:Date, y=:Temperature, Geom.point, Theme(discrete_highlight_color=c->nothing))
```

```@example wooster
df = copy(data)
df[!,:Temperature] = -data[:,:Temperature]
filter!(row -> month(row.Date) ∈ (1,2,11,12), df)
plot(df, x=:Date, y=:Temperature, Geom.point)
```

### Declustering the threshold exceedances

```@example wooster
threshold = -10
cluster = getcluster(df[:,:Temperature], -10, runlength=4)
nothing #hide
```

```@repl wooster
typeof(cluster)
```

### GPD parameters estimation

Let's first identify the threshold exceedances:
```@example rain
threshold = 30.0
df = filter(row -> row.Rainfall > threshold, data)
first(df, 5)
```

Get the exceedances above the threshold:
```@example rain
df[!,:Rainfall] =  df[!,:Rainfall] .- threshold
rename!(df, :Rainfall => :Exceedance)
first(df, 5)
```

Generalized Pareto parameter estimation by maximum likelihood:
```@repl rain
fm = gpfit(df, :Exceedance)
```

!!! note

    The function returns the estimates of the log-scale parameter $\phi = \log \sigma$.


### Return level estimation

With the [`ThresholdExceedance`](@ref) structure, the [`returnlevel`](@ref) function requires several arguments to calculate the *T*-year return level:
- the threshold value;
- the number of total observation (below and above the threshold);
- the number of observations per year;
- the return period *T*;
- the confidence level for computing the confidence interval.
The function uses the Peaks-Over-Threshold model definition (Coles, 2001, Chapter 4) for computing the *T*-year return level.

For the rainfall example, the 100-year return level can be estimated as follows:

```@repl rain
r = returnlevel(fm, threshold, size(data,1), 365, 100, .95)
```

where the value can be accessed with
```@repl rain
r.value
```

and where the corresponding confidence interval can be accessed with
```@repl rain
r.cint
```

!!! note

    In this example of a stationary model, the function returns a unit dimension vector for the return level and a vector containing only one vector for the confidence interval. The reason is that the function always returns the same type in the stationary and non-stationary case. The function is therefore [type-stable](https://docs.julialang.org/en/v1/manual/performance-tips/index.html#Write-%22type-stable%22-functions-1) allowing better performance of code execution.  

To get the scalar return level in the stationary case, the following command can be used:
```@repl rain
r.value[]
```

To get the scalar confidence interval in the stationary case, the following command can be used:
```@repl rain
r.cint[]
```

### Probability weighted moments estimation  

Probability weighted moments estimation of the GEV parameters can also be performed by using the [`gevfitpwm`](@ref) function. All the methods also apply to the `pwmEVA` object.

```@repl rain
fm = gpfitpwm(df, :Exceedance)
```

### Bayesian estimation

Bayesian estimation of the GEV parameters can also be performed by using the [`gevfitbayes`](@ref) function. All the methods also apply to the `BayesianEVA object.

```@repl rain
fm = gpfitbayes(df, :Exceedance)
```







## Model for non-stationary data
Coles(2001, Chapter 6)
