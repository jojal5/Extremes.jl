# Extreme value analysis package for Julia.


[![Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Build status](https://github.com/jojal5/Extremes.jl/workflows/CI/badge.svg)](https://github.com/jojal5/Extremes.jl/actions)
[![codecov](https://codecov.io/gh/jojal5/Extremes.jl/branch/master/graph/badge.svg?token=7UGVMF0ENE)](https://codecov.io/gh/jojal5/Extremes.jl)
[![documentation stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://jojal5.github.io/Extremes.jl/stable/)
[![documentation latest](https://img.shields.io/badge/docs-latest-blue.svg)](https://jojal5.github.io/Extremes.jl/dev/)



## Installation

The following **julia** command will install the package:

```julia
julia> Pkg.add("Extremes")
```

See the [Package Documentation](https://jojal5.github.io/Extremes.jl/dev/) for details and examples on how to use the package.

See also the related paper by Jalbert *et al.* (2024, to appear) describing the package, along with the [Jupyter notebook file](docs/src/JOSS/JOSS.ipynb), which replicates the results and the figures. The notebook can be consulted online *via* nbviewer following this [link](https://nbviewer.org/github/jojal5/Extremes.jl/blob/dev/docs/src/JOSS/JOSS.ipynb).

Reference: 
Jalbert, J., Farmer, M., Gobeil, G. and Roy, P. (2023). Extremes.jl: Extreme Value Analysis in Julia. Provisionally accepted in *Journal of Statistical Software*.


## Data
The datasets that are available through this package are the datasets referenced in *An Introduction to Statistical Modeling of Extreme Values* by Stuart Coles.

They were obtained using the R package `ismev`.  
https://www.rdocumentation.org/packages/ismev/  
