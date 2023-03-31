
# Block Maxima Model

The stationary [`BlockMaxima`](@ref) model is illustrated using the annual maximum sea-levels recorded at Port Pirie in South Australia from 1923 to 1987, studied by Coles (2001) in Chapter 3. The annual maxima are assumed **independent and identically distributed**.

```@setup portpirie
using Extremes, DataFrames, Distributions, Gadfly
```

The *Extremes.jl* package supports maximum likelihood inference, Bayesian inference and inference based on the probability weigthed moments. For the GEV parameter estimation, the following functions can be used:
- [`gevfitpwm`](@ref): estimation with the probability weighted moments;
- [`gevfit`](@ref): maximum likelihood estimation;
- [`gevfitbayes`](@ref): Bayesian estimation.


!!! note "Log-scale paremeter"

    These functions shows the estimate of the log-scale parameter $\phi = \log \sigma$ instead of the scale parameter.


## Load the data

Loading the annual maximum sea-levels at Port Pirie:
```@example portpirie
data = Extremes.dataset("portpirie")
first(data,5)
```

The annual maxima can be shown as a function of the year using the Gadfly package:
```@example portpirie
set_default_plot_size(12cm, 8cm)
plot(data, x=:Year, y=:SeaLevel, Geom.line)
```

## Maximum likelihood inference


### GEV parameter estimation

The GEV parameter estimation with maximum likelihood is performed with the [`gevfit`](@ref) function:

```@repl portpirie
fm = gevfit(data, :SeaLevel)
```

!!! note
    In this example, the [`gevfit`](@ref) function is called using the data *DataFrame* structure as the first argument. The function can also be called using the vector of maxima as the first argument, *e.g.* `gevfit(data[:,:SeaLevel])`.


The vector of the parameter estimates (location scale and shape) can be extracted with the function [`params`](@ref):
```@repl portpirie
params(fm)
```
, the location parameter with the function [`location`](@ref):
```@repl portpirie
location(fm)
```

the scale parameter with the function [`Extremes.scale`](@ref):
```@repl portpirie
scale(fm)
```
and the shape parameter with the function [`shape`](@ref):
```@repl portpirie
shape(fm)
```

!!! note "Type-stable function"

    These functions return a unit dimension vector for the return level and a vector containing only one vector for the confidence interval. The reason is that the functions always return the same type in the stationary and non-stationary case. The functions are therefore [type-stable](https://docs.julialang.org/en/v1/manual/performance-tips/index.html#Write-%22type-stable%22-functions-1) allowing better performance of code execution.  



The approximate covariance matrix of the parameter estimates can be obtained with the  [`parametervar`](@ref) function:
```@repl portpirie
parametervar(fm)
```

Confidence intervals for the parameters are obtained with the [`cint`](@ref) function:
```@repl portpirie
cint(fm)
```

For instance, the shape parameter 95% confidence interval is as follows:
```@repl portpirie
cint(fm)[3]
```

### Diagnostic plots

Several diagnostic plots for assessing the accuracy of the GEV model fitted to the Port Pirie data are can be shown with the [`diagnosticplots`](@ref) function:

```@example portpirie
set_default_plot_size(21cm ,16cm)
diagnosticplots(fm)
```

The diagnostic plots consist in the probability plot (upper left panel), the quantile plot (upper right panel), the density plot (lower left panel) and the return level plot (lower right panel). These plots can be displayed separately using respectively the [`probplot`](@ref), [`qqplot`](@ref), [`histplot`](@ref) and [`returnlevelplot`](@ref) functions.


### Return level estimation

*T*-year return level estimate can be obtained using the [`returnlevel`](@ref) function. For example, the 100-year return level for the Port Pirie block maxima model is computed as follows:
```@repl portpirie
r = returnlevel(fm, 100)
```

The return level can be accessed as follows:
```@repl portpirie
r.value
```

The corresponding confidence interval can be computed with the [`cint`](@ref) function:
```@repl portpirie
c = cint(r)
```

To get the scalar return level in the stationary case, the following command can be used:
```@repl portpirie
r.value[]
```

To get the scalar confidence interval in the stationary case, the following command can be used:
```@repl portpirie
c[]
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

## Inference for the Gumbel distribution

The inference for the Gumbel distribution is also provided for modeling the series of block maxima. The Gumbel distribution is a sub-family of the GEV distribution when the shape parameter ``\xi = 0``. The library provides functions analogous to those for the GEV distribution for the adjustment of the parameters of the Gumbel distribution:
- by maximum likelihood [`gumbelfit`](@ref);
- under the Bayesian paradigm [`gumbelfitbayes`](@ref);
- by the method of moments [`gumbelfitpwm`](@ref).
The inference for the fitted model is performed with the same methods defined previously for the GEV distribution, notably with the [`cint`](@ref) and [`returnlevel`](@ref) methods. Diagnostic plot methods [`diagnosticplots`](@ref) are also implemented, such as [`probplot`](@ref), [`qqplot`](@ref), [`histplot`](@ref) and [`returnlevelplot`](@ref).

!!! note "Gumbel distribution for modelling block maxima"
    As extreme-value statisticians, we advocate avoiding the use of the Gumbel distribution for modelling the block maxima. This is because the choice of family is made with the data at hand, and when extrapolating to large quantiles, i.e. larger than the range of the data, the uncertainty associated with this choice is not taken into account. If the data suggest that the Gumbel family is the best one, this does not imply that the other families are not plausible. In applications, the confidence intervals on the shape parameter are often wide, representing the difficulty of discriminating the tail behavior using only the limited number of data. Therefore, we plead for the use of the GEV distribution for the block maxima model without choosing a subfamily such as the Gumbel. As Coles (2001) also argued in Page 64, this is "... the safest option is to accept there is uncertainty about the value of the shape parameter ... and to prefer the inference based on the GEV model. The larger measures of uncertainty generated by the GEV model then provide a more realistic quantification of the genuine uncertainties involved in model extrapolation".

### Example on the annual maximum sea-levels recorded at Port Pirie

The Gumbel distribution parameter estimation with maximum likelihood can be performed with the [`gumbelfit`](@ref) function:

```@repl portpirie
fm = gumbelfit(data, :SeaLevel)
```

The approximate covariance matrix of the parameter estimates can be obtained with the  [`parametervar`](@ref) function:
```@repl portpirie
parametervar(fm)
```

Confidence intervals for the parameters are obtained with the [`cint`](@ref) function:
```@repl portpirie
cint(fm)
```

The diagnostic plots for assessing the accuracy of the Gumbel model can be shown with the [`diagnosticplots`](@ref) function:

```@example portpirie
set_default_plot_size(21cm ,16cm)
diagnosticplots(fm)
```

The 100-year return level for the Port Pirie block maxima model is computed with the [`returnlevel`](@ref) function  as follows:
```@repl portpirie
r = returnlevel(fm, 100)
```

The return level can be accessed as follows:
```@repl portpirie
r.value
```

The corresponding confidence interval can be computed with the [`cint`](@ref) function:
```@repl portpirie
c = cint(r)
```
