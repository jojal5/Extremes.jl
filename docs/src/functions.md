# Functions

## Parameter estimation

```@autodocs
Modules = [Extremes]
Private = false
Order = [:function]
Pages = [
    "src/parameterestimation/maximumlikelihood.jl",
    "src/parameterestimation/probabilityweightedmoment.jl",
    "src/parameterestimation/bayesian.jl",
    "src/parameterestimation/maximumlikelihood/maximumlikelihood_gev.jl",
    "src/parameterestimation/maximumlikelihood/maximumlikelihood_gumbel.jl",
    "src/parameterestimation/maximumlikelihood/maximumlikelihood_gp.jl",
    "src/parameterestimation/probabilityweightedmoment/probabilityweightedmoment_gev.jl",
    "src/parameterestimation/probabilityweightedmoment/probabilityweightedmoment_gumbel.jl",
    "src/parameterestimation/probabilityweightedmoment/probabilityweightedmoment_gp.jl",
    "src/parameterestimation/bayesian/bayesian_gev.jl",
    "src/parameterestimation/bayesian/bayesian_gumbel.jl",
    "src/parameterestimation/bayesian/bayesian_gp.jl"
    ]
```

## Methods on fitted models

```@autodocs
Modules = [Extremes]
Private = false
Order = [:function]
Pages = [
    "src/structures/cluster.jl",
    "src/structures/dataitem.jl",
    "src/structures/AbstractExtremeValueModel.jl",
    "src/structures/AbstractFittedExtremeValueModel.jl",
    "src/structures/returnlevel.jl",
    "src/structures/dataitem/variable.jl",
    "src/structures/dataitem/variablestd.jl",
    "src/structures/AbstractExtremeValueModel/blockmaxima.jl",
    "src/structures/AbstractExtremeValueModel/thresholdexceedance.jl",
    "src/structures/AbstractFittedExtremeValueModel/bayesianAbstractExtremeValueModel.jl",
    "src/structures/AbstractFittedExtremeValueModel/maximumlikelihoodAbstractExtremeValueModel.jl",
    "src/structures/AbstractFittedExtremeValueModel/pwmAbstractExtremeValueModel.jl"
    ]
```

## Diagnostic plots

```@autodocs
Modules = [Extremes]
Private = false
Order = [:function]
Pages = [
    "src/validationplots/plots.jl",
    "src/validationplots/plots_data.jl",
    "src/validationplots/plots_std_data.jl"
    ]
```


## Types

```@autodocs
Modules = [Extremes]
Private = false
Order = [:type]
```
