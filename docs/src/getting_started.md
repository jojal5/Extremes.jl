# Tutorial

This tutorial shows the functionalities of the package *Extremes.jl*. It is illustrated by reproducing all the results shown by Coles (2001) in [An Introduction to Statistical Modeling of Extreme
 Values](http://www.springer.com/us/book/9781852334598).

 Before executing this tutorial, make sure to install the following packages:
 ```@repl
using Pkg
Pkg.add("Extremes")

# For using the DataFrame type.
Pkg.add("DataFrames")

# For using probability distribution objects
Pkg.add("Distributions")

# For plotting.
Pkg.add("Gadfly")

# or Bayesian inference.
Pkg.add("Mamba")

# importing those packages
using Extremes, DataFrames, Distributions, Gadfly, Mamba
```

## Model for stationary block maxima
Coles(2001, Chapter 3)

### Data loading

Loading the annual maximum sea-levels at Port Pirie:
```@repl
data = load("portpirie")
```

Plotting the data using the Gadfly package:
```@repl
plot(data, x=:Year, y=:SeaLevel, Geom.line)
```
