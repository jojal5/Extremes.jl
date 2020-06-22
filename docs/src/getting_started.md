# Tutorial

This tutorial shows the functionalities of the package *Extremes.jl*. They are illustrated by reproducing the results shown by Coles (2001) in [An Introduction to Statistical Modeling of Extreme
 Values](http://www.springer.com/us/book/9781852334598).

 Before executing this tutorial, make sure to have installed the following packages:
- Extremes (of course)
- DataFrames (for using the DataFrame type)
- Distributions (for using probability distribution objects)
- Gadfly (for plotting)
- Mamba (for performing Bayesian paradigm)

Import those packages:
 ```@repl
using Extremes, DataFrames, Distributions, Gadfly, Mamba
```

## Model for stationary block maxima
Coles(2001, Chapter 3)

### Port Pirie example

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

In this case, the function returns a MaximumLikelihoodEVA object which contains:
- the structure name indicating notably the estimation method (the maximum likelihood in this example);
- the statistical model (the stationary block maxima model in this example);
- the location, log-scale and shape parameter estimates respectively in the vector $ θ̂ $.


## Model for stationary threshold exceedances
Coles(2001, Chapter 4)

## Model for dependent data
Coles(2001, Chapter 5)

## Model for non-stationary data
Coles(2001, Chapter 6)
