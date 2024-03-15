# Extreme value analysis package for Julia.


[![Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Build status](https://github.com/jojal5/Extremes.jl/workflows/CI/badge.svg)](https://github.com/jojal5/Extremes.jl/actions)
[![codecov](https://codecov.io/gh/jojal5/Extremes.jl/branch/master/graph/badge.svg?token=7UGVMF0ENE)](https://codecov.io/gh/jojal5/Extremes.jl)
[![documentation stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://jojal5.github.io/Extremes.jl/stable/)
[![documentation latest](https://img.shields.io/badge/docs-latest-blue.svg)](https://jojal5.github.io/Extremes.jl/dev/)


The `Extremes.jl` package provides exhaustive, high-performance functions by leveraging the multiple-dispatch capabilities in **Julia** for the analysis of extreme values. In particular, the package implements statistical models for
- block maxima;
- peaks-over-threshold;
along with several methods for the generalized extreme value and generalized Pareto distributions used in extreme value theory. 

Additionally, the package offers various parameter estimation methods, such as 
- probability-weighted moments;
- maximum likelihood;
- Bayesian estimation. 
It also includes tools for handling dependence in excesses over a threshold and methods for managing nonstationary models. Inference for extreme quantiles is available for both stationary and nonstationary models, along with diagnostic figures to assess the goodness of fit of the model to the data.

## Documentation

See the [Package Documentation](https://jojal5.github.io/Extremes.jl/dev/) for details and examples on how to use the package.

Additionally, refer to the related paper by Jalbert *et al.* (2024, to appear) which describes the package, along with the accompanying [Jupyter notebook file](docs/src/JOSS/JOSS.ipynb) that replicates the results and figures. The notebook can be viewed online *via* nbviewer through this [link](https://nbviewer.org/github/jojal5/Extremes.jl/blob/dev/docs/src/JOSS/JOSS.ipynb).

Reference: 
Jalbert, J., Farmer, M., Gobeil, G. and Roy, P. (2023). Extremes.jl: Extreme Value Analysis in Julia. Provisionally accepted in *Journal of Statistical Software*.

## Installation

The following **julia** command will install the package:

```julia
julia> Pkg.add("Extremes")
```

See the [Package Documentation](https://jojal5.github.io/Extremes.jl/dev/) for details and examples on how to use the package.

See also the related paper by Jalbert *et al.* (2024, to appear) describing the package, along with the [Jupyter notebook file](docs/src/JOSS/JOSS.ipynb), which replicates the results and the figures. The notebook can be consulted online *via* nbviewer following this [link](https://nbviewer.org/github/jojal5/Extremes.jl/blob/dev/docs/src/JOSS/JOSS.ipynb).


## Data
The datasets that are available through this package are the datasets referenced in *An Introduction to Statistical Modeling of Extreme Values* by Stuart Coles.

They were obtained using the R package `ismev`.  
https://www.rdocumentation.org/packages/ismev/  
