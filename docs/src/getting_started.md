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
using Extremes, DataFrames, Distributions, Gadfly
```





## Model for stationary block maxima

### Port Pirie example

This section concerns the annual maximum sea-levels recorded at Port Pirie, South Australia, from 1923 to 1987. This dataset were studied by Coles(2001) in Chapter 3.

```@setup portpirie
using Extremes, DataFrames, Distributions, Gadfly
```

#### Load the data

Loading the annual maximum sea-levels at Port Pirie:
```@example portpirie
data = load("portpirie")
first(data,5)
```

Plotting the data using the Gadfly package:
```@example portpirie
plot(data, x=:Year, y=:SeaLevel, Geom.line)
```

#### GEV parameters estimation

In this example, the Generalized Extreme Value distribution is fitted by maximum likelihood to the annual maximum sea-levels at Port-Pirie.

The data have been loaded in a *DataFrame*. The function `gevfit` can be called directly using the dataframe as the first argument and the data column symbol as the second argument as follows:

```@repl portpirie
fm = gevfit(data, :SeaLevel)
```

The function [`gevfit`](@ref) returns a MaximumLikelihoodEVA object which contains:
- the structure name indicating in particular the estimation method (maximum likelihood in this example);
- the statistical model (the stationary block maxima model in this example);
- the location, log-scale and shape parameter estimates respectively in the vector $ θ̂ $.

!!! note

    The function returns the estimates of the log-scale parameter $\phi = \log \sigma$.

#### Diagnostics plots

    TODO

#### Return level estimation

*T*-year return level estimate can be obtained using the function [`returnlevel`](@ref) on a `fittedEVA` object. The first argument is the fitted model, the second is the return period in years and the last one is the confidence level for computing the confidence interval.

For example, the 100-year return level for the Port Pirie data and the corresponding 95% confidence interval can be obtained with this commands:

```@repl portpirie
r = returnlevel(fm, 100, .95)
```

where the value can be accessed with
```@repl portpirie
r.value
```

and where the corresponding confidence interval can be accessed with
```@repl portpirie
r.cint
```

!!! note

    In this example of a stationary model, the function returns a unit dimension vector for the return level and a vector containing only one vector for the confidence interval. The reason is that the function always returns the same type in the stationary and non-stationary case. The function is therefore [type-stable](https://docs.julialang.org/en/v1/manual/performance-tips/index.html#Write-%22type-stable%22-functions-1) allowing better performance of code execution.  

To get the scalar return level in the stationary case, the following command can be used:
```@repl portpirie
r.value[]
```

To get the scalar confidence interval in the stationary case, the following command can be used:
```@repl portpirie
r.cint[]
```

#### Probability weighted moments estimation  

Probability weighted moments estimation of the GEV parameters can also be performed by using the [`gevfitpwm`](@ref) function. All the methods also apply to the `pwmEVA` object.

```@repl portpirie
fm = gevfitpwm(data[:,:SeaLevel])
```

#### Bayesian estimation

Bayesian estimation of the GEV parameters can also be performed by using the [`gevfitbayes`](@ref) function. All the methods also apply to the `BayesianEVA object.

```@repl portpirie
fm = gevfitbayes(data[:,:SeaLevel])
```






## Model for stationary threshold exceedances

The data of this section come from Chapter 4 of Coles (2001) and correspond to the daily rainfall accumulations at a location in south-west England from 1914 to 1962.

```@setup rain
using Extremes, DataFrames, Distributions, Gadfly, Dates
```

#### Load the data

Loading the daily rainfall at a location in South-England:

```@example rain
data = load("rain")
x = collect(Date(1914,1,1):Day(1):Date(1961,12,30))
data[!,:Date] = x
select!(data, [:Date, :Rainfall])
first(data,5)
```

Plotting the data using the Gadfly package:
```@example rain
plot(data, x=:Date, y=:Rainfall, Geom.point, Theme(discrete_highlight_color=c->nothing))
```

#### Threshold selection

TODO

#### GPD parameters estimation

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


#### Return level estimation

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

#### Probability weighted moments estimation  

Probability weighted moments estimation of the GEV parameters can also be performed by using the [`gevfitpwm`](@ref) function. All the methods also apply to the `pwmEVA` object.

```@repl rain
fm = gpfitpwm(df, :Exceedance)
```

#### Bayesian estimation

Bayesian estimation of the GEV parameters can also be performed by using the [`gevfitbayes`](@ref) function. All the methods also apply to the `BayesianEVA object.

```@repl rain
fm = gpfitbayes(df, :Exceedance)
```



## Model for dependent data
Coles(2001, Chapter 5)

## Model for non-stationary data
Coles(2001, Chapter 6)
